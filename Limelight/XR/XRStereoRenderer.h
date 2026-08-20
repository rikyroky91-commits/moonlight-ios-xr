//
//  XRStereoRenderer.h
//  Moonlight XR
//

@import Foundation;
@import QuartzCore;
@import CoreVideo;

NS_ASSUME_NONNULL_BEGIN

/// Converte il flusso video 2D in un frame half-SBS con profondita' stimata
/// dall'iPhone, destinato a occhiali AR in modalita' 3D.
///
/// Con lo stereo spento disegna una singola vista non deformata, che e' anche
/// il modo giusto di guardare contenuto gia' SBS prodotto dal PC.
@interface XRStereoRenderer : NSObject

/// Restituisce nil se Metal non e' disponibile o le pipeline non compilano.
/// Il modello di profondita' e' opzionale: se manca, il renderer funziona
/// comunque e resta in modalita' piatta.
- (nullable instancetype)initWithLayer:(CAMetalLayer *)layer;

/// Se NO, disegna una sola vista a piena larghezza senza alcuna elaborazione.
@property (nonatomic) BOOL stereoEnabled;

/// Disparita' massima come frazione della larghezza immagine. E' il comando di
/// "quanto 3D": oltre il 2% la maggior parte delle persone fatica a fondere le
/// due immagini. Default 0.012.
@property (nonatomic) float maxDisparity;

/// Profondita' normalizzata (0 = piu' vicino, 1 = piu' lontano) che cade sul
/// piano dello schermo. Alzarla spinge la scena dietro lo schermo, abbassarla
/// la fa uscire verso l'osservatore. Default 0.35.
@property (nonatomic) float convergence;

/// Zoom sul contenuto per lasciare margine ai bordi. Default 1.0.
@property (nonatomic) float zoom;

/// YES quando la stima di profondita' e' attiva e ha prodotto almeno un
/// risultato: fino ad allora lo stereo disegna due viste identiche.
@property (nonatomic, readonly) BOOL depthReady;

/// Durata dell'ultima inferenza in millisecondi, per la diagnostica.
@property (nonatomic, readonly) double lastInferenceMs;

/// Da chiamare per ogni frame decodificato. Non blocca in attesa della GPU.
- (void)renderPixelBuffer:(CVPixelBufferRef)pixelBuffer;

@end

NS_ASSUME_NONNULL_END
