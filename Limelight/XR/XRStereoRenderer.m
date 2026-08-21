//
//  XRStereoRenderer.m
//  Moonlight XR
//

#import "XRStereoRenderer.h"
#import "XRDepthEstimator.h"
#import "Logger.h"

@import Metal;
@import simd;

// Risoluzione della mappa di disparita'. Meta' di 1080p: la profondita' e' un
// segnale a bassa frequenza spaziale e alzarla non migliora il risultato, mentre
// il costo del warp cresce linearmente.
static const NSUInteger kDisparityWidth = 960;
static const NSUInteger kDisparityHeight = 540;

static const NSUInteger kHistogramBins = 64;

// Deve restare allineato alle costanti dello shader.
static const NSUInteger kMotionCandidates = 187;
static const NSUInteger kMotionThreads = 64;

// Frazioni dell'istogramma scartate agli estremi per la normalizzazione: senza
// questo margine un riflesso speculare o una porzione di cielo sposterebbero
// l'intera scala di profondita'.
static const float kLowPercentile = 0.02f;
static const float kHighPercentile = 0.98f;

// Quanto lentamente si muove l'intervallo di profondita'. Lento e' meglio: un
// intervallo che insegue la scena fa "pompare" tutta la profondita'.
static const float kRangeSmoothing = 0.02f;

typedef struct { vector_float2 sourceScale; } XRPrepareUniforms;

typedef struct {
    float alphaMin;
    float alphaMax;
    float motionGain;
    float rawLag;
} XRStabilizeUniforms;

typedef struct {
    float depthLo;
    float depthHi;
    float convergence;
    float maxDisparity;
    float edgeSigma;
} XRDisparityUniforms;

typedef struct {
    float eyeSign;
    float zoom;
    float searchRadius;
    float occlusionBias;
    int   searchTaps;
} XRWarpUniforms;

@implementation XRStereoRenderer {
    CAMetalLayer* _layer;
    id<MTLDevice> _device;
    id<MTLCommandQueue> _queue;

    id<MTLComputePipelineState> _preparePipeline;
    id<MTLComputePipelineState> _motionSearchPipeline;
    id<MTLComputePipelineState> _motionPickPipeline;
    id<MTLComputePipelineState> _stabilizePipeline;
    id<MTLComputePipelineState> _disparityPipeline;
    id<MTLRenderPipelineState> _warpPipeline;
    id<MTLRenderPipelineState> _plainPipeline;

    CVMetalTextureCacheRef _textureCache;
    XRDepthEstimator* _depth;

    // Ping-pong per il filtro temporale: il frame corrente legge il precedente.
    id<MTLTexture> _stableDepth[2];
    id<MTLTexture> _lumaSmall[2];
    NSUInteger _pingPong;

    id<MTLTexture> _disparityTexture;
    id<MTLBuffer> _histogramBuffer;
    id<MTLBuffer> _motionScores;
    id<MTLBuffer> _motionVector;

    // Intervallo di profondita' filtrato, aggiornato dal completion handler.
    float _depthLo;
    float _depthHi;
    BOOL _rangeInitialized;

    dispatch_semaphore_t _inFlight;
}

- (nullable instancetype)initWithLayer:(CAMetalLayer *)layer {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _device = MTLCreateSystemDefaultDevice();
    if (_device == nil) {
        Log(LOG_E, @"XR: Metal non disponibile su questo dispositivo");
        return nil;
    }

    _layer = layer;
    _layer.device = _device;
    _layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    _layer.framebufferOnly = YES;
    _layer.presentsWithTransaction = NO;

    _queue = [_device newCommandQueue];
    id<MTLLibrary> library = [_device newDefaultLibrary];
    if (_queue == nil || library == nil) {
        Log(LOG_E, @"XR: command queue o default.metallib mancanti");
        return nil;
    }

    if (![self buildPipelinesWithLibrary:library]) {
        return nil;
    }

    if (CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, _device, NULL, &_textureCache) != kCVReturnSuccess) {
        Log(LOG_E, @"XR: CVMetalTextureCacheCreate fallita");
        return nil;
    }

    [self buildIntermediateTextures];

    // Il modello e' opzionale: senza, restiamo utilizzabili in modalita' piatta.
    _depth = [[XRDepthEstimator alloc] initWithDevice:_device];
    if (_depth == nil) {
        Log(LOG_W, @"XR: stima di profondita' non disponibile, solo modalita' piatta");
    }

    _inFlight = dispatch_semaphore_create(2);
    _maxDisparity = 0.012f;
    _convergence = 0.35f;
    // Un margine di ritaglio e' indispensabile: il warp sposta anche i pixel di
    // bordo, e senza margine il bordo stesso diventa una linea ondulata che
    // segue il profilo di profondita' della scena.
    _zoom = 1.05f;
    _stereoEnabled = YES;

    Log(LOG_I, @"XR: renderer stereo pronto su %@", _device.name);
    return self;
}

- (BOOL)buildPipelinesWithLibrary:(id<MTLLibrary>)library {
    NSError* error = nil;

    NSArray<NSString*>* computeNames = @[@"xr_prepare_input", @"xr_motion_search",
                                        @"xr_motion_pick", @"xr_depth_stabilize",
                                        @"xr_disparity_build"];
    NSMutableArray* computePipelines = [NSMutableArray array];
    for (NSString* name in computeNames) {
        id<MTLFunction> fn = [library newFunctionWithName:name];
        if (fn == nil) {
            Log(LOG_E, @"XR: kernel '%@' assente dalla libreria", name);
            return NO;
        }
        id<MTLComputePipelineState> pipeline = [_device newComputePipelineStateWithFunction:fn error:&error];
        if (pipeline == nil) {
            Log(LOG_E, @"XR: kernel '%@' non compilato: %@", name, error);
            return NO;
        }
        [computePipelines addObject:pipeline];
    }
    _preparePipeline = computePipelines[0];
    _motionSearchPipeline = computePipelines[1];
    _motionPickPipeline = computePipelines[2];
    _stabilizePipeline = computePipelines[3];
    _disparityPipeline = computePipelines[4];

    id<MTLFunction> vertexFn = [library newFunctionWithName:@"xr_vertex"];
    if (vertexFn == nil) {
        Log(LOG_E, @"XR: vertex shader assente");
        return NO;
    }

    NSArray<NSString*>* fragmentNames = @[@"xr_fragment_warp", @"xr_fragment_plain"];
    NSMutableArray* renderPipelines = [NSMutableArray array];
    for (NSString* name in fragmentNames) {
        id<MTLFunction> fragmentFn = [library newFunctionWithName:name];
        if (fragmentFn == nil) {
            Log(LOG_E, @"XR: fragment shader '%@' assente", name);
            return NO;
        }
        MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
        desc.vertexFunction = vertexFn;
        desc.fragmentFunction = fragmentFn;
        desc.colorAttachments[0].pixelFormat = _layer.pixelFormat;

        id<MTLRenderPipelineState> pipeline = [_device newRenderPipelineStateWithDescriptor:desc error:&error];
        if (pipeline == nil) {
            Log(LOG_E, @"XR: pipeline '%@' non compilata: %@", name, error);
            return NO;
        }
        [renderPipelines addObject:pipeline];
    }
    _warpPipeline = renderPipelines[0];
    _plainPipeline = renderPipelines[1];

    return YES;
}

- (void)buildIntermediateTextures {
    for (NSUInteger i = 0; i < 2; i++) {
        MTLTextureDescriptor* depthDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                               width:kXRDepthWidth
                                                              height:kXRDepthHeight
                                                           mipmapped:NO];
        depthDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        depthDesc.storageMode = MTLStorageModePrivate;
        _stableDepth[i] = [_device newTextureWithDescriptor:depthDesc];

        MTLTextureDescriptor* lumaDesc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                               width:kXRDepthWidth
                                                              height:kXRDepthHeight
                                                           mipmapped:NO];
        lumaDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        lumaDesc.storageMode = MTLStorageModePrivate;
        _lumaSmall[i] = [_device newTextureWithDescriptor:lumaDesc];
    }

    MTLTextureDescriptor* dispDesc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                           width:kDisparityWidth
                                                          height:kDisparityHeight
                                                       mipmapped:NO];
    dispDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    dispDesc.storageMode = MTLStorageModePrivate;
    _disparityTexture = [_device newTextureWithDescriptor:dispDesc];

    _histogramBuffer = [_device newBufferWithLength:kHistogramBins * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];

    _motionScores = [_device newBufferWithLength:kMotionCandidates * sizeof(float)
                                         options:MTLResourceStorageModePrivate];
    // Parte da zero: nessuno spostamento finche' non c'e' un frame precedente.
    vector_float2 zero = (vector_float2){0.0f, 0.0f};
    _motionVector = [_device newBufferWithBytes:&zero
                                         length:sizeof(zero)
                                        options:MTLResourceStorageModePrivate];
}

- (void)dealloc {
    if (_textureCache != NULL) {
        CFRelease(_textureCache);
    }
}

- (double)lastInferenceMs {
    return _depth.lastInferenceMs;
}

#pragma mark - Texture dai CVPixelBuffer

- (nullable id<MTLTexture>)textureFromBuffer:(CVPixelBufferRef)pixelBuffer
                                       plane:(size_t)plane
                                 pixelFormat:(MTLPixelFormat)format {
    size_t width = CVPixelBufferGetWidthOfPlane(pixelBuffer, plane);
    size_t height = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane);

    CVMetalTextureRef cvTexture = NULL;
    if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, _textureCache, pixelBuffer,
                                                  NULL, format, width, height, plane,
                                                  &cvTexture) != kCVReturnSuccess || cvTexture == NULL) {
        return nil;
    }

    id<MTLTexture> texture = CVMetalTextureGetTexture(cvTexture);
    CFRelease(cvTexture);
    return texture;
}

- (nullable id<MTLTexture>)bgraTextureFromBuffer:(CVPixelBufferRef)pixelBuffer {
    CVMetalTextureRef cvTexture = NULL;
    if (CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault, _textureCache, pixelBuffer,
                                                  NULL, MTLPixelFormatBGRA8Unorm,
                                                  CVPixelBufferGetWidth(pixelBuffer),
                                                  CVPixelBufferGetHeight(pixelBuffer), 0,
                                                  &cvTexture) != kCVReturnSuccess || cvTexture == NULL) {
        return nil;
    }

    id<MTLTexture> texture = CVMetalTextureGetTexture(cvTexture);
    CFRelease(cvTexture);
    return texture;
}

#pragma mark - Rendering

- (void)renderPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    if (pixelBuffer == NULL || _plainPipeline == nil) {
        return;
    }

    // Se la GPU e' indietro si scarta: un frame vecchio mostrato in ritardo e'
    // peggio di un frame saltato.
    if (dispatch_semaphore_wait(_inFlight, DISPATCH_TIME_NOW) != 0) {
        return;
    }

    id<MTLTexture> lumaTex = [self textureFromBuffer:pixelBuffer plane:0 pixelFormat:MTLPixelFormatR8Unorm];
    id<MTLTexture> chromaTex = [self textureFromBuffer:pixelBuffer plane:1 pixelFormat:MTLPixelFormatRG8Unorm];
    id<CAMetalDrawable> drawable = (lumaTex && chromaTex) ? [_layer nextDrawable] : nil;
    if (drawable == nil) {
        dispatch_semaphore_signal(_inFlight);
        return;
    }

    id<MTLCommandBuffer> commandBuffer = [_queue commandBuffer];

    const BOOL wantDepth = self.stereoEnabled && _depth != nil;
    id<MTLTexture> depthTexture = wantDepth ? [_depth latestDepthTexture] : nil;
    const BOOL useWarp = (depthTexture != nil);

    CVPixelBufferRef modelInput = NULL;
    if (wantDepth) {
        modelInput = [_depth acquireInputBuffer];
        if (modelInput != NULL) {
            [self encodePrepare:commandBuffer input:modelInput luma:lumaTex chroma:chromaTex];
        }
        if (useWarp) {
            [self encodeDepth:commandBuffer rawDepth:depthTexture luma:lumaTex];
        }
    }

    [self encodeDraw:commandBuffer drawable:drawable luma:lumaTex chroma:chromaTex useWarp:useWarp];

    __block dispatch_semaphore_t inFlight = _inFlight;
    __block CVPixelBufferRef submitBuffer = NULL;
    if (modelInput != NULL) {
        submitBuffer = (CVPixelBufferRef)CFRetain(modelInput);
    }

    __weak XRStereoRenderer* weakSelf = self;
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull buffer) {
        XRStereoRenderer* strongSelf = weakSelf;
        if (strongSelf != nil) {
            if (useWarp) {
                // L'istogramma e' completo solo ora: leggerlo qui evita del tutto
                // di sincronizzare CPU e GPU durante il frame.
                [strongSelf updateDepthRangeFromHistogram];
            }
            if (submitBuffer != NULL) {
                [strongSelf->_depth submitInputBuffer:submitBuffer];
            }
        }
        if (submitBuffer != NULL) {
            CFRelease(submitBuffer);
        }
        dispatch_semaphore_signal(inFlight);
    }];

    [commandBuffer presentDrawable:drawable];
    [commandBuffer commit];

    if (useWarp) {
        _pingPong ^= 1;
    }
}

- (void)encodePrepare:(id<MTLCommandBuffer>)commandBuffer
                input:(CVPixelBufferRef)modelInput
                 luma:(id<MTLTexture>)lumaTex
               chroma:(id<MTLTexture>)chromaTex {
    id<MTLTexture> rgbTarget = [self bgraTextureFromBuffer:modelInput];
    if (rgbTarget == nil) {
        return;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_preparePipeline];
    [encoder setTexture:lumaTex atIndex:0];
    [encoder setTexture:chromaTex atIndex:1];
    [encoder setTexture:rgbTarget atIndex:2];
    [encoder setTexture:_lumaSmall[_pingPong] atIndex:3];

    XRPrepareUniforms uniforms = { .sourceScale = (vector_float2){1.0f, 1.0f} };
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:0];

    [self dispatch:encoder pipeline:_preparePipeline width:kXRDepthWidth height:kXRDepthHeight];
    [encoder endEncoding];
}

- (void)encodeDepth:(id<MTLCommandBuffer>)commandBuffer
           rawDepth:(id<MTLTexture>)rawDepth
               luma:(id<MTLTexture>)lumaFull {
    // L'istogramma va azzerato prima di ogni accumulo.
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    [blit fillBuffer:_histogramBuffer range:NSMakeRange(0, _histogramBuffer.length) value:0];
    [blit endEncoding];

    const NSUInteger current = _pingPong;
    const NSUInteger previous = _pingPong ^ 1;

    // Stima dello spostamento dell'inquadratura fra il frame precedente e
    // questo, valutando in parallelo tutti gli spostamenti candidati.
    id<MTLComputeCommandEncoder> motion = [commandBuffer computeCommandEncoder];
    [motion setComputePipelineState:_motionSearchPipeline];
    [motion setTexture:_lumaSmall[current] atIndex:0];
    [motion setTexture:_lumaSmall[previous] atIndex:1];
    [motion setBuffer:_motionScores offset:0 atIndex:0];
    [motion dispatchThreadgroups:MTLSizeMake(kMotionCandidates, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kMotionThreads, 1, 1)];

    [motion setComputePipelineState:_motionPickPipeline];
    [motion setBuffer:_motionScores offset:0 atIndex:0];
    [motion setBuffer:_motionVector offset:0 atIndex:1];
    [motion dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [motion endEncoding];

    id<MTLComputeCommandEncoder> stabilize = [commandBuffer computeCommandEncoder];
    [stabilize setComputePipelineState:_stabilizePipeline];
    [stabilize setTexture:rawDepth atIndex:0];
    [stabilize setTexture:_stableDepth[previous] atIndex:1];
    [stabilize setTexture:_lumaSmall[current] atIndex:2];
    [stabilize setTexture:_lumaSmall[previous] atIndex:3];
    [stabilize setTexture:_stableDepth[current] atIndex:4];
    [stabilize setBuffer:_histogramBuffer offset:0 atIndex:0];

    // Filtraggio molto piu' deciso di prima. L'alpha adattivo era tarato per il
    // video: in un gioco la telecamera muove tutta l'inquadratura, ogni pixel
    // risulta "in movimento", il filtro si spegneva sempre e restava lo
    // sfarfallio nudo, percepito come un ondeggiamento subacqueo.
    XRStabilizeUniforms stabilizeUniforms = {
        .alphaMin = 0.05f,
        .alphaMax = 0.30f,
        .motionGain = 1.5f,
        // L'inferenza dura circa un frame e il risultato e' disponibile due
        // frame dopo quello da cui e' partita.
        .rawLag = 1.5f,
    };
    [stabilize setBytes:&stabilizeUniforms length:sizeof(stabilizeUniforms) atIndex:1];
    [stabilize setBuffer:_motionVector offset:0 atIndex:2];
    [self dispatch:stabilize pipeline:_stabilizePipeline width:kXRDepthWidth height:kXRDepthHeight];
    [stabilize endEncoding];

    id<MTLComputeCommandEncoder> disparity = [commandBuffer computeCommandEncoder];
    [disparity setComputePipelineState:_disparityPipeline];
    [disparity setTexture:_stableDepth[current] atIndex:0];
    [disparity setTexture:_lumaSmall[current] atIndex:1];
    [disparity setTexture:lumaFull atIndex:2];
    [disparity setTexture:_disparityTexture atIndex:3];

    XRDisparityUniforms disparityUniforms = {
        .depthLo = _rangeInitialized ? _depthLo : 0.0f,
        .depthHi = _rangeInitialized ? _depthHi : 1.0f,
        .convergence = self.convergence,
        .maxDisparity = self.maxDisparity,
        .edgeSigma = 0.12f,
    };
    [disparity setBytes:&disparityUniforms length:sizeof(disparityUniforms) atIndex:0];
    [self dispatch:disparity pipeline:_disparityPipeline width:kDisparityWidth height:kDisparityHeight];
    [disparity endEncoding];
}

- (void)encodeDraw:(id<MTLCommandBuffer>)commandBuffer
          drawable:(id<CAMetalDrawable>)drawable
              luma:(id<MTLTexture>)lumaTex
            chroma:(id<MTLTexture>)chromaTex
           useWarp:(BOOL)useWarp {
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = drawable.texture;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:useWarp ? _warpPipeline : _plainPipeline];
    [encoder setFragmentTexture:lumaTex atIndex:0];
    [encoder setFragmentTexture:chromaTex atIndex:1];
    if (useWarp) {
        [encoder setFragmentTexture:_disparityTexture atIndex:2];
    }

    const double fullWidth = drawable.texture.width;
    const double fullHeight = drawable.texture.height;
    const double halfWidth = fullWidth / 2.0;

    const BOOL stereo = self.stereoEnabled;
    const int viewCount = stereo ? 2 : 1;
    const float eyeSigns[2] = { 1.0f, -1.0f };

    for (int eye = 0; eye < viewCount; eye++) {
        MTLViewport viewport = {
            .originX = stereo ? (eye * halfWidth) : 0.0,
            .originY = 0.0,
            .width = stereo ? halfWidth : fullWidth,
            .height = fullHeight,
            .znear = 0.0,
            .zfar = 1.0
        };
        [encoder setViewport:viewport];

        XRWarpUniforms uniforms = {
            .eyeSign = stereo ? eyeSigns[eye] : 0.0f,
            .zoom = self.zoom,
            .searchRadius = self.maxDisparity,
            .occlusionBias = 0.12f,
            // Piu' campioni significa passo di ricerca piu' fine, quindi
            // tolleranza piu' stretta sulle corrispondenze e meno bave.
            .searchTaps = 32,
        };
        [encoder setFragmentBytes:&uniforms length:sizeof(uniforms) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    }

    [encoder endEncoding];
}

- (void)dispatch:(id<MTLComputeCommandEncoder>)encoder
        pipeline:(id<MTLComputePipelineState>)pipeline
           width:(NSUInteger)width
          height:(NSUInteger)height {
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = MAX(pipeline.maxTotalThreadsPerThreadgroup / w, (NSUInteger)1);
    MTLSize threadgroup = MTLSizeMake(w, h, 1);
    MTLSize groups = MTLSizeMake((width + w - 1) / w, (height + h - 1) / h, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threadgroup];
}

/// Ricava dall'istogramma un intervallo di profondita' robusto e lo insegue
/// lentamente. E' quel che impedisce alla scena di "pompare" quando entra in
/// campo un oggetto molto vicino o molto lontano.
- (void)updateDepthRangeFromHistogram {
    const uint32_t* bins = (const uint32_t*)_histogramBuffer.contents;

    uint32_t total = 0;
    for (NSUInteger i = 0; i < kHistogramBins; i++) {
        total += bins[i];
    }
    if (total == 0) {
        return;
    }

    const uint32_t lowTarget = (uint32_t)(total * kLowPercentile);
    const uint32_t highTarget = (uint32_t)(total * kHighPercentile);

    float lo = 0.0f;
    float hi = 1.0f;
    uint32_t running = 0;
    BOOL lowFound = NO;
    for (NSUInteger i = 0; i < kHistogramBins; i++) {
        running += bins[i];
        if (!lowFound && running >= lowTarget) {
            lo = (float)i / (float)kHistogramBins;
            lowFound = YES;
        }
        if (running >= highTarget) {
            hi = (float)(i + 1) / (float)kHistogramBins;
            break;
        }
    }

    if (hi - lo < 0.05f) {
        // Scena quasi piatta: allargare evita di amplificare il rumore fino a
        // trasformarlo in profondita' inventata.
        float mid = (lo + hi) * 0.5f;
        lo = MAX(0.0f, mid - 0.025f);
        hi = MIN(1.0f, mid + 0.025f);
    }

    if (!_rangeInitialized) {
        _depthLo = lo;
        _depthHi = hi;
        _rangeInitialized = YES;
    }
    else {
        _depthLo += (lo - _depthLo) * kRangeSmoothing;
        _depthHi += (hi - _depthHi) * kRangeSmoothing;
    }
}

- (BOOL)depthReady {
    return _depth != nil && [_depth latestDepthTexture] != nil;
}

@end
