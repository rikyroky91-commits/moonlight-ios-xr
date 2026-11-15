# 🥽 Moonlight XR — real-time 2D→3D game streaming for AR glasses, computed on the iPhone

![iOS](https://img.shields.io/badge/iOS-Objective--C-000000) ![Metal](https://img.shields.io/badge/GPU-Metal-8A8A8A) ![Core ML](https://img.shields.io/badge/AI-Core%20ML%20%2F%20Neural%20Engine-34AADC) ![License](https://img.shields.io/badge/license-GPLv3-blue) ![CI](https://img.shields.io/badge/IPA-GitHub%20Actions-2088FF)

**A fork of [Moonlight iOS](https://github.com/moonlight-stream/moonlight-ios) that turns any PC game stream into
stereoscopic 3D, live and on-device, for USB-C AR glasses (RayNeo Air, XREAL, Viture).**
No changes on the PC: no ReShade, no depth-buffer injectors, nothing required from the game.

### How it works
```
decoder → CVPixelBuffer (NV12)
        → Depth Anything V2 Small on the Neural Engine (~17 ms)
        → adaptive temporal filter → robust histogram normalisation
        → luma-guided bilateral upsampling
        → occlusion-aware gather warp, one view per eye
        → half-SBS 1920×1080 on the external display
```
Inference runs on the **Neural Engine with the GPU explicitly excluded**, because the GPU is busy with the
warp in the same milliseconds.

### What makes the 3D comfortable
| Problem | Without a fix | Solution |
|---|---|---|
| Temporal instability | depth "breathes" | per-pixel adaptive alpha driven by luma motion |
| Scale drift | scene pumps when a close object enters | 64-bin histogram, robust 2/98 percentiles, slow tracking |
| Blurry edges | halos around objects | joint bilateral upsampling guided by colour |
| Occlusions | black holes behind edges | gather warp where the nearest surface wins |

### Try it
No Mac needed: GitHub Actions builds and publishes an unsigned IPA on every push.
**[Download the latest IPA](https://github.com/rikyroky91-commits/moonlight-ios-xr/releases/latest)** and sideload
it with AltStore or SideStore. Plug in the glasses, switch them to 3D mode and tap **3D**.

### Tech
Objective-C · VideoToolbox · Metal compute shaders · Core ML (Depth Anything V2) · Neural Engine · GitHub Actions macOS runners

---

## 🇮🇹 In breve

Fork di Moonlight iOS che converte lo streaming di gioco 2D in **3D stereo direttamente sull'iPhone** e lo manda
agli occhiali AR via USB-C. Nessuna modifica sul PC. Documentazione completa delle scelte tecniche in
[README-XR.md](README-XR.md).

---

Credits: Moonlight iOS by the Moonlight team (GPLv3, original README in
[README-moonlight-upstream.md](README-moonlight-upstream.md)); external display support from
[PR #411](https://github.com/moonlight-stream/moonlight-ios/pull/411) by mattrussell7; depth model
[Depth Anything V2 Small](https://huggingface.co/apple/coreml-depth-anything-v2-small) (Apache 2.0).
