//
//  XRDepthEstimator.h
//  Moonlight XR
//

@import Foundation;
@import CoreVideo;
@import Metal;

NS_ASSUME_NONNULL_BEGIN

/// Dimensioni fisse imposte dal modello Depth Anything V2 Small convertito da
/// Apple: non ammette forme alternative.
extern const size_t kXRDepthWidth;   // 518
extern const size_t kXRDepthHeight;  // 392

/// Stima la profondita' monoculare del frame video su Neural Engine.
///
/// L'inferenza gira su una coda dedicata e non blocca mai il decoder: se un
/// frame arriva mentre il modello sta ancora lavorando viene scartato. Il
/// renderer usa sempre l'ultimo risultato disponibile, che nella pratica e'
/// vecchio di uno o due frame.
@interface XRDepthEstimator : NSObject

/// Restituisce nil se il modello non e' nel bundle o non si carica.
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device;

/// Fornisce un CVPixelBuffer BGRA 518x392 da riempire col frame ridimensionato.
/// Restituisce nil se l'estimatore e' occupato: in quel caso salta la conversione
/// e non sprecare tempo GPU.
- (nullable CVPixelBufferRef)acquireInputBuffer CF_RETURNS_NOT_RETAINED;

/// Invia per l'inferenza un buffer ottenuto da acquireInputBuffer.
- (void)submitInputBuffer:(CVPixelBufferRef)buffer;

/// Ultima mappa di profondita' come texture r16Float 518x392, o nil finche' non
/// arriva il primo risultato. Puo' essere letta da qualsiasi thread.
- (nullable id<MTLTexture>)latestDepthTexture;

/// Durata dell'ultima inferenza in millisecondi, per la diagnostica.
@property (atomic, readonly) double lastInferenceMs;

@end

NS_ASSUME_NONNULL_END
