# Moonlight iOS XR

Fork di [moonlight-ios](https://github.com/moonlight-stream/moonlight-ios) che converte
il flusso di gioco 2D in stereo 3D **sull'iPhone**, e lo manda a occhiali AR collegati
via USB-C (RayNeo Air, XREAL, Viture) in modalità side-by-side.

Nessuna modifica al PC: niente ReShade, niente depth buffer injector, nessun requisito
sul gioco.

## Come funziona

Il video decodificato non passa più per `AVSampleBufferDisplayLayer` ma per una
`VTDecompressionSession`, che restituisce i pixel. Da lì la pipeline è interamente
Metal e Core ML:

```
decoder → CVPixelBuffer (NV12)
        → downscale a 518x392 → Depth Anything V2 Small su Neural Engine (~17 ms)
        → filtro temporale ad alpha adattivo
        → normalizzazione su istogramma (percentili 2/98, inseguiti lentamente)
        → upsampling bilaterale guidato dalla luma
        → warp gather con risoluzione delle occlusioni, una vista per occhio
        → frame half-SBS 1920x1080 sul display esterno
```

L'inferenza gira su Neural Engine con la GPU **esplicitamente esclusa**: la GPU sta
facendo il warp negli stessi millisecondi, e farli competere farebbe crollare entrambi.

### Le quattro scelte che contano

Una conversione ingenua produce un 3D che stanca in pochi minuti. I problemi affrontati:

| Problema | Effetto senza rimedio | Soluzione adottata |
|---|---|---|
| Instabilità temporale | la profondità "respira" | alpha adattivo per pixel, guidato dal movimento della luminanza |
| Deriva di scala | la scena pompa quando entra un oggetto vicino | istogramma a 64 bin, percentili robusti, inseguimento lento |
| Bordi sfocati | aloni attorno agli oggetti | upsampling bilaterale congiunto guidato dal colore |
| Occlusioni | buchi neri dietro i bordi | warp gather: fra i candidati vince la superficie più vicina |

## Uso

Senza occhiali collegati si comporta esattamente come Moonlight originale.

Collegando gli occhiali via USB-C il video migra sul display esterno a 1920x1080 nativi
(960 per occhio) e sul telefono compare un tasto **3D**. Sugli occhiali va attivata la
modalità 3D: sui RayNeo Air 4 Pro si tiene premuto il tasto R1.

Con il tasto 3D **spento** l'app mostra il frame 1:1, che è anche il modo corretto di
guardare contenuto già SBS generato dal PC.

Impostazioni consigliate: **1080p60** (il pannello degli occhiali non va oltre) e
**HDR disattivato** (la pipeline lavora a 8 bit).

### Limiti noti

- Gli HUD di gioco vengono interpretati come parte della scena e finiscono a profondità
  arbitrarie. Alzare la convergenza li riporta vicino al piano dello schermo.
- La profondità è indietro di uno o due frame rispetto al colore: durante i movimenti
  rapidi si nota come un leggero slittamento.

## Compilazione

Non serve un Mac: il workflow `.github/workflows/build-unsigned-ipa.yml` compila su
runner macOS e pubblica un IPA non firmato, da installare con AltStore o SideStore.

## Licenze

Codice GPLv3, come il progetto originale. Il modello
[Depth Anything V2 Small](https://huggingface.co/apple/coreml-depth-anything-v2-small)
è Apache 2.0. Il supporto al display esterno viene dalla
[PR #411](https://github.com/moonlight-stream/moonlight-ios/pull/411) di mattrussell7,
mai integrata a monte.
