//
//  XRShaders.metal
//  Moonlight XR
//
//  Pipeline di conversione 2D -> stereo, in quattro passaggi:
//
//    1. xr_prepare_input    video NV12 -> BGRA 518x392 per Core ML, piu' una
//                           luma di pari dimensione usata dal filtro temporale
//    2. xr_depth_stabilize  fonde la profondita' nuova con quella precedente e
//                           accumula l'istogramma per la normalizzazione
//    3. xr_disparity_build  porta la profondita' a mezza risoluzione con un
//                           upsampling guidato dal colore e la converte in
//                           disparita' in unita' UV
//    4. xr_vertex/xr_fragment_warp  warp gather con gestione delle occlusioni,
//                           una vista per occhio
//

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- strutture

struct XRPrepareUniforms {
    float2 sourceScale;   // riservato per ritagli futuri, oggi 1,1
};

struct XRStabilizeUniforms {
    // Peso minimo e massimo del frame nuovo. La stima monocolare oscilla da un
    // frame all'altro anche su scene ferme: senza questo filtro l'immagine
    // "respira" ed e' la causa numero uno di affaticamento.
    float alphaMin;
    float alphaMax;
    // Quanto una variazione di luminanza spinge alpha verso alphaMax: dove il
    // colore cambia c'e' movimento vero e la profondita' nuova va creduta.
    float motionGain;
};

struct XRDisparityUniforms {
    float depthLo;         // estremi robusti dell'istogramma, gia' filtrati
    float depthHi;
    float convergence;     // profondita' normalizzata che finisce sul piano schermo
    float maxDisparity;    // disparita' massima in unita' UV della sorgente
    float edgeSigma;       // tolleranza di colore dell'upsampling guidato
};

struct XRWarpUniforms {
    float eyeSign;         // +1 occhio sinistro, -1 destro
    float zoom;
    float searchRadius;    // in unita' UV, pari a maxDisparity
    int   searchTaps;
};

struct XRVertexOut {
    float4 position [[position]];
    float2 uv;
};

constant uint kHistogramBins = 64;

// ------------------------------------------------------ utilita' di colore

static inline float3 xr_yuv_to_rgb(float y, float2 cbcr)
{
    // BT.709 video range: Y 16-235, CbCr 16-240 su 8 bit.
    y = (y - 16.0 / 255.0) * (255.0 / 219.0);
    float cb = (cbcr.x - 128.0 / 255.0) * (255.0 / 224.0);
    float cr = (cbcr.y - 128.0 / 255.0) * (255.0 / 224.0);

    float3 rgb;
    rgb.r = y + 1.5748 * cr;
    rgb.g = y - 0.1873 * cb - 0.4681 * cr;
    rgb.b = y + 1.8556 * cb;
    return saturate(rgb);
}

// ------------------------------------------- 1. preparazione input Core ML

kernel void xr_prepare_input(texture2d<float, access::sample> lumaTex    [[texture(0)]],
                             texture2d<float, access::sample> chromaTex  [[texture(1)]],
                             texture2d<float, access::write>  rgbOut     [[texture(2)]],
                             texture2d<float, access::write>  lumaSmall  [[texture(3)]],
                             constant XRPrepareUniforms&      u          [[buffer(0)]],
                             uint2                            gid        [[thread_position_in_grid]])
{
    if (gid.x >= rgbOut.get_width() || gid.y >= rgbOut.get_height()) {
        return;
    }

    constexpr sampler smp(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(rgbOut.get_width(), rgbOut.get_height());
    uv *= u.sourceScale;

    float  y    = lumaTex.sample(smp, uv).r;
    float2 cbcr = chromaTex.sample(smp, uv).rg;
    float3 rgb  = xr_yuv_to_rgb(y, cbcr);

    // Il modello vuole BGRA: Core ML legge i canali in quest'ordine.
    rgbOut.write(float4(rgb.b, rgb.g, rgb.r, 1.0), gid);
    lumaSmall.write(float4(y, 0.0, 0.0, 1.0), gid);
}

// ------------------------------------------ 2. stabilizzazione temporale

kernel void xr_depth_stabilize(texture2d<float, access::read>  rawDepth   [[texture(0)]],
                               texture2d<float, access::read>  prevStable [[texture(1)]],
                               texture2d<float, access::read>  lumaCur    [[texture(2)]],
                               texture2d<float, access::read>  lumaPrev   [[texture(3)]],
                               texture2d<float, access::write> stableOut  [[texture(4)]],
                               device atomic_uint*             histogram  [[buffer(0)]],
                               constant XRStabilizeUniforms&   u          [[buffer(1)]],
                               uint2                           gid        [[thread_position_in_grid]])
{
    if (gid.x >= stableOut.get_width() || gid.y >= stableOut.get_height()) {
        return;
    }

    float current = rawDepth.read(gid).r;
    float previous = prevStable.read(gid).r;

    // Alpha adattivo per pixel: fermo si filtra molto, in movimento si segue il
    // dato nuovo. Un alpha fisso obbligherebbe a scegliere fra sfarfallio sulle
    // scene statiche e scie sulle panoramiche.
    float motion = abs(lumaCur.read(gid).r - lumaPrev.read(gid).r);
    float alpha = clamp(u.alphaMin + motion * u.motionGain, u.alphaMin, u.alphaMax);

    // Al primo frame previous vale 0 e non va mescolato.
    float stable = (previous > 0.0) ? mix(previous, current, alpha) : current;
    stableOut.write(float4(stable, 0.0, 0.0, 1.0), gid);

    // Istogramma per la normalizzazione robusta: usare min/max renderebbe la
    // scala ostaggio di un singolo pixel anomalo.
    uint bin = uint(clamp(stable, 0.0, 0.999) * float(kHistogramBins));
    atomic_fetch_add_explicit(&histogram[bin], 1u, memory_order_relaxed);
}

// ------------------------------- 3. upsampling guidato e calcolo disparita'

kernel void xr_disparity_build(texture2d<float, access::sample> stableDepth [[texture(0)]],
                               texture2d<float, access::sample> lumaSmall   [[texture(1)]],
                               texture2d<float, access::sample> lumaFull    [[texture(2)]],
                               texture2d<float, access::write>  disparity   [[texture(3)]],
                               constant XRDisparityUniforms&    u           [[buffer(0)]],
                               uint2                            gid         [[thread_position_in_grid]])
{
    const uint width = disparity.get_width();
    const uint height = disparity.get_height();
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    constexpr sampler smp(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(width, height);

    float guide = lumaFull.sample(smp, uv).r;
    float2 texel = 1.0 / float2(stableDepth.get_width(), stableDepth.get_height());

    // Upsampling bilaterale congiunto: i vicini che somigliano al pixel guida
    // pesano di piu'. E' quello che tiene i bordi della profondita' incollati
    // ai bordi degli oggetti, invece di produrre aloni.
    float sumWeight = 0.0;
    float sumDepth = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float2 offset = float2(dx, dy) * texel;
            float d = stableDepth.sample(smp, uv + offset).r;
            float l = lumaSmall.sample(smp, uv + offset).r;

            float colorDist = (l - guide) / max(u.edgeSigma, 0.001);
            float spatial = (dx == 0 && dy == 0) ? 1.0 : 0.6;
            float weight = spatial * exp(-colorDist * colorDist);

            sumDepth += d * weight;
            sumWeight += weight;
        }
    }

    float depth = (sumWeight > 0.0) ? (sumDepth / sumWeight) : stableDepth.sample(smp, uv).r;

    // Normalizzazione sull'intervallo robusto, poi conversione in disparita'
    // centrata sul piano di convergenza: cio' che sta a "convergence" cade sullo
    // schermo, il resto esce o rientra.
    float range = max(u.depthHi - u.depthLo, 0.001);
    float normalized = clamp((depth - u.depthLo) / range, 0.0, 1.0);
    float signedDepth = normalized - u.convergence;

    disparity.write(float4(signedDepth * u.maxDisparity, 0.0, 0.0, 1.0), gid);
}

// ------------------------------------------------------- 4. warp per occhio

vertex XRVertexOut xr_vertex(uint vid [[vertex_id]])
{
    const float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    const float2 uvs[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };

    XRVertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}

fragment float4 xr_fragment_warp(XRVertexOut in [[stage_in]],
                                 texture2d<float> lumaTex   [[texture(0)]],
                                 texture2d<float> chromaTex [[texture(1)]],
                                 texture2d<float> disparity [[texture(2)]],
                                 constant XRWarpUniforms& u [[buffer(0)]])
{
    constexpr sampler smp(filter::linear, address::clamp_to_edge);

    float2 uv = (in.uv - 0.5) / max(u.zoom, 0.001) + 0.5;

    // Warp gather con consapevolezza delle occlusioni.
    //
    // Per il pixel di uscita cerchiamo quale pixel sorgente ci finisce sopra
    // una volta spostato della propria disparita'. Piu' candidati possono
    // mappare sullo stesso punto: e' esattamente cio' che accade su un bordo di
    // occlusione. Vince quello con disparita' maggiore, cioe' la superficie
    // piu' vicina all'osservatore, che nella realta' e' quella che copre
    // l'altra. Un warp forward lascerebbe invece buchi neri sui bordi.
    float bestDisparity = 0.0;
    float bestError = 1e9;
    bool  found = false;

    const int taps = max(u.searchTaps, 2);
    const float step = (2.0 * u.searchRadius) / float(taps - 1);

    for (int i = 0; i < taps; i++) {
        float candidateX = uv.x - u.searchRadius + step * float(i);
        float d = disparity.sample(smp, float2(candidateX, uv.y)).r;

        // Dove finisce questo pixel sorgente nella vista di questo occhio
        float mapped = candidateX + u.eyeSign * d;
        float error = abs(mapped - uv.x);

        // Accettiamo solo candidati che cadono entro mezzo passo di ricerca,
        // altrimenti si prenderebbero corrispondenze inventate.
        if (error < step) {
            bool better = !found
                || (d > bestDisparity + 1e-6)
                || (abs(d - bestDisparity) <= 1e-6 && error < bestError);
            if (better) {
                bestDisparity = d;
                bestError = error;
                found = true;
            }
        }
    }

    // Nessun candidato: e' una zona scoperta dietro un bordo. Ripiegare sulla
    // disparita' locale la riempie con lo sfondo circostante, che e' molto meno
    // visibile di un buco.
    if (!found) {
        bestDisparity = disparity.sample(smp, uv).r;
    }

    float2 sourceUV = float2(uv.x - u.eyeSign * bestDisparity, uv.y);

    if (sourceUV.x < 0.0 || sourceUV.x > 1.0 || sourceUV.y < 0.0 || sourceUV.y > 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float  y    = lumaTex.sample(smp, sourceUV).r;
    float2 cbcr = chromaTex.sample(smp, sourceUV).rg;
    return float4(xr_yuv_to_rgb(y, cbcr), 1.0);
}

// Percorso mono: nessuna profondita', nessun warp. Serve quando il 3D e'
// spento e quando la stima non e' ancora pronta.
fragment float4 xr_fragment_plain(XRVertexOut in [[stage_in]],
                                  texture2d<float> lumaTex   [[texture(0)]],
                                  texture2d<float> chromaTex [[texture(1)]],
                                  constant XRWarpUniforms& u [[buffer(0)]])
{
    constexpr sampler smp(filter::linear, address::clamp_to_edge);
    float2 uv = (in.uv - 0.5) / max(u.zoom, 0.001) + 0.5;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    float  y    = lumaTex.sample(smp, uv).r;
    float2 cbcr = chromaTex.sample(smp, uv).rg;
    return float4(xr_yuv_to_rgb(y, cbcr), 1.0);
}
