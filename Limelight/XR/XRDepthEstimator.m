//
//  XRDepthEstimator.m
//  Moonlight XR
//

#import "XRDepthEstimator.h"
#import "Logger.h"

@import CoreML;

#include <stdatomic.h>

const size_t kXRDepthWidth = 518;
const size_t kXRDepthHeight = 392;

static NSString* const kModelResourceName = @"DepthAnythingV2SmallF16";
static NSString* const kInputFeatureName = @"image";
static NSString* const kOutputFeatureName = @"depth";

@implementation XRDepthEstimator {
    MLModel* _model;
    dispatch_queue_t _inferenceQueue;
    CVPixelBufferPoolRef _inputPool;
    CVMetalTextureCacheRef _textureCache;

    // Un solo frame in volo: se il modello e' occupato scartiamo, invece di
    // accodare lavoro che sarebbe comunque obsoleto quando finisce.
    _Atomic(int) _busy;

    // Protegge la coppia buffer/texture piu' recente, scritta dalla coda di
    // inferenza e letta dal thread di rendering.
    NSLock* _resultLock;
    CVPixelBufferRef _latestDepthBuffer;
    CVMetalTextureRef _latestDepthCVTexture;
    id<MTLTexture> _latestDepthTexture;
}

/// Xcode normalmente compila il .mlpackage in .mlmodelc dentro il bundle. Se per
/// qualsiasi ragione non lo fa, il pacchetto e' comunque presente come risorsa e
/// lo compiliamo noi al primo avvio, conservando il risultato: la compilazione
/// costa qualche secondo e non ha senso ripeterla a ogni sessione.
- (nullable NSURL*)resolveModelURL {
    NSBundle* bundle = [NSBundle mainBundle];
    NSFileManager* fm = [NSFileManager defaultManager];

    NSURL* compiled = [bundle URLForResource:kModelResourceName withExtension:@"mlmodelc"];
    if (compiled != nil) {
        return compiled;
    }

    NSURL* package = [bundle URLForResource:kModelResourceName withExtension:@"mlpackage"];
    if (package == nil) {
        Log(LOG_E, @"XR: modello %@ assente dal bundle in ogni forma", kModelResourceName);
        return nil;
    }

    NSURL* supportDir = [[fm URLsForDirectory:NSApplicationSupportDirectory
                                    inDomains:NSUserDomainMask] firstObject];
    NSURL* cached = [supportDir URLByAppendingPathComponent:
                     [kModelResourceName stringByAppendingPathExtension:@"mlmodelc"]];
    if (cached != nil && [fm fileExistsAtPath:cached.path]) {
        return cached;
    }

    Log(LOG_I, @"XR: compilazione del modello al primo avvio, puo' richiedere qualche secondo");

    NSError* error = nil;
    NSURL* temporary = [MLModel compileModelAtURL:package error:&error];
    if (temporary == nil) {
        Log(LOG_E, @"XR: compilazione del modello fallita: %@", error);
        return nil;
    }

    if (cached == nil) {
        return temporary;
    }

    [fm createDirectoryAtURL:supportDir withIntermediateDirectories:YES attributes:nil error:NULL];
    [fm removeItemAtURL:cached error:NULL];
    if (![fm moveItemAtURL:temporary toURL:cached error:&error]) {
        Log(LOG_W, @"XR: cache del modello non riuscita, uso la copia temporanea: %@", error);
        return temporary;
    }
    return cached;
}

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    NSURL* url = [self resolveModelURL];
    if (url == nil) {
        return nil;
    }

    MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
    // Deliberatamente senza GPU: la GPU serve al warp stereo, e lasciarli
    // competere farebbe crollare entrambi. Sull'ANE l'inferenza e' in pratica
    // gratuita rispetto al budget di frame.
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

    NSError* error = nil;
    _model = [MLModel modelWithContentsOfURL:url configuration:config error:&error];
    if (_model == nil) {
        Log(LOG_E, @"XR: caricamento modello fallito: %@", error);
        return nil;
    }

    CVReturn cvret = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, device, NULL, &_textureCache);
    if (cvret != kCVReturnSuccess) {
        Log(LOG_E, @"XR: texture cache per la profondita' fallita: %d", cvret);
        return nil;
    }

    NSDictionary* pixelAttrs = @{
        (id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
        (id)kCVPixelBufferWidthKey: @(kXRDepthWidth),
        (id)kCVPixelBufferHeightKey: @(kXRDepthHeight),
        (id)kCVPixelBufferMetalCompatibilityKey: @YES,
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
    };
    NSDictionary* poolAttrs = @{ (id)kCVPixelBufferPoolMinimumBufferCountKey: @(3) };

    cvret = CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                    (__bridge CFDictionaryRef)poolAttrs,
                                    (__bridge CFDictionaryRef)pixelAttrs,
                                    &_inputPool);
    if (cvret != kCVReturnSuccess) {
        Log(LOG_E, @"XR: pool di input fallito: %d", cvret);
        return nil;
    }

    _inferenceQueue = dispatch_queue_create("com.moonlight-stream.xr.depth",
                                            dispatch_queue_attr_make_with_qos_class(
                                                DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
    _resultLock = [[NSLock alloc] init];

    Log(LOG_I, @"XR: stima di profondita' pronta (%zux%zu, Neural Engine)",
        kXRDepthWidth, kXRDepthHeight);
    return self;
}

- (void)dealloc {
    [self releaseLatestLocked:NO];
    if (_inputPool != NULL) {
        CVPixelBufferPoolRelease(_inputPool);
    }
    if (_textureCache != NULL) {
        CFRelease(_textureCache);
    }
}

- (nullable CVPixelBufferRef)acquireInputBuffer {
    // Nessun controllo di occupazione qui: il renderer riempie comunque questo
    // buffer a ogni frame, perche' lo stesso passaggio produce anche la luma
    // ridotta che serve al filtro temporale. E' submitInputBuffer a scartare.
    CVPixelBufferRef buffer = NULL;
    CVReturn cvret = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, _inputPool, &buffer);
    if (cvret != kCVReturnSuccess) {
        return NULL;
    }

    // Il chiamante non possiede il buffer: lo restituisce con submitInputBuffer,
    // che se ne assume il rilascio.
    return (CVPixelBufferRef)CFAutorelease(buffer);
}

- (void)submitInputBuffer:(CVPixelBufferRef)buffer {
    if (buffer == NULL) {
        return;
    }

    int expected = 0;
    if (!atomic_compare_exchange_strong(&_busy, &expected, 1)) {
        return;
    }

    CVPixelBufferRef retained = (CVPixelBufferRef)CFRetain(buffer);

    dispatch_async(_inferenceQueue, ^{
        CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
        [self runInference:retained];
        self->_lastInferenceMs = (CFAbsoluteTimeGetCurrent() - start) * 1000.0;
        CFRelease(retained);
        atomic_store(&self->_busy, 0);
    });
}

- (void)runInference:(CVPixelBufferRef)input {
    NSError* error = nil;

    MLFeatureValue* value = [MLFeatureValue featureValueWithPixelBuffer:input];
    MLDictionaryFeatureProvider* provider =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{ kInputFeatureName: value }
                                                          error:&error];
    if (provider == nil) {
        Log(LOG_E, @"XR: costruzione input fallita: %@", error);
        return;
    }

    id<MLFeatureProvider> result = [_model predictionFromFeatures:provider error:&error];
    if (result == nil) {
        Log(LOG_E, @"XR: inferenza fallita: %@", error);
        return;
    }

    CVPixelBufferRef depthBuffer = [result featureValueForName:kOutputFeatureName].imageBufferValue;
    if (depthBuffer == NULL) {
        Log(LOG_E, @"XR: output '%@' assente o non e' un'immagine", kOutputFeatureName);
        return;
    }

    // L'output e' grayscale float16: entra in Metal come r16Float senza copie.
    CVMetalTextureRef cvTexture = NULL;
    CVReturn cvret = CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
                                                               _textureCache,
                                                               depthBuffer,
                                                               NULL,
                                                               MTLPixelFormatR16Float,
                                                               CVPixelBufferGetWidth(depthBuffer),
                                                               CVPixelBufferGetHeight(depthBuffer),
                                                               0,
                                                               &cvTexture);
    if (cvret != kCVReturnSuccess || cvTexture == NULL) {
        Log(LOG_E, @"XR: texture di profondita' fallita: %d", cvret);
        return;
    }

    [_resultLock lock];
    [self releaseLatestLocked:YES];
    _latestDepthBuffer = (CVPixelBufferRef)CFRetain(depthBuffer);
    _latestDepthCVTexture = cvTexture;
    _latestDepthTexture = CVMetalTextureGetTexture(cvTexture);
    [_resultLock unlock];
}

/// Rilascia buffer e texture correnti. Il lock e' gia' preso dal chiamante
/// quando `locked` e' YES.
- (void)releaseLatestLocked:(BOOL)locked {
    if (!locked) {
        [_resultLock lock];
    }
    if (_latestDepthCVTexture != NULL) {
        CFRelease(_latestDepthCVTexture);
        _latestDepthCVTexture = NULL;
    }
    if (_latestDepthBuffer != NULL) {
        CFRelease(_latestDepthBuffer);
        _latestDepthBuffer = NULL;
    }
    _latestDepthTexture = nil;
    if (!locked) {
        [_resultLock unlock];
    }
}

- (nullable id<MTLTexture>)latestDepthTexture {
    [_resultLock lock];
    id<MTLTexture> texture = _latestDepthTexture;
    [_resultLock unlock];
    return texture;
}

@end
