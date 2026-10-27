# MetalFX Denoising against our network, in the frame

2026-10-27. Apple M3 Max, macOS 26.6.2 (25G83), Xcode 26.2, AC power, the
GPU idle. The network: the committed frame at `bd4d8b0` (Milestone 6's
finished frame, change 3's network through Metal 4's machine-learning
path). MetalFX: a scratch prototype on the same commit, outside the
repository and not committed.

**Decided by the user, 2026-10-27: keep our network.** MetalFX is faster
at every size but loses what the scene is built around: the caustics
inside the glass and the fireflies' pinpoints in the mirrors.

## 1. The question

The Neural Engine note (`2026-10-26-neural-engine-denoiser.md`, section 4)
named MetalFX Denoising as Apple's production answer to the frame's
denoiser, not measured. Apple's guidance for rendering (WWDC26 session
359) puts networks on the GPU at three levels; MetalFX Denoising is the
first, Apple's own denoising upscaler, replacing our trained network
outright. This note puts it in the frame with every input Apple lists, at
every size in play, against our network: frame rates and frames side by
side.

## 2. How MetalFX was wired

**The API: MetalFX's older one, not Metal 4's.** `MTL4FXTemporalDenoisedScaler`
cannot be made at our sizes: its creation (`newTemporalDenoisedScalerWithDevice:compiler:`)
aborts inside MPSGraph, `MPSGraphExecutable.mm:3467: failed assertion
'Incompatible shape for parameter at index 0'`. A standalone program with
none of our code, creating the scaler each way:

| Input size | `MTLFXTemporalDenoisedScaler` | `MTL4FXTemporalDenoisedScaler` |
|---|---|---|
| 960 x 540 | made | made |
| 1728 x 1117 (3456 x 2234 at 2x) | made | aborts |
| 1152 x 745 (3456 x 2234 at 3x) | made | aborts |
| 480 x 270 | made | aborts |

Depth as R32Float or Depth32Float, color and output in 16 or 32 bits, auto
exposure on or off, synchronous initialization or not, and nearby sizes
(1728 x 1118, 1728 x 1120, 480 x 272, 480 x 268, 482 x 270) change
nothing: only 960 x 540 is made. The cause is inside Apple's framework
and not known. No report of this assertion was found; a forum thread
reports a different Metal 4 MetalFX failure (an encode that writes
nothing), with bug numbers Apple has not confirmed. Not filed with Apple
(the user's call).

So the scaler runs on a classic command queue of its own, sharing the
frame's residency set. The image half of the frame is split around it:
the part before it is committed to the Metal 4 overlap queue, which
signals a shared event; the classic queue's command buffer waits for it,
encodes MetalFX and signals a second event; the part after it (depth of
field, tone map, presentation) waits for that. The frame's GPU time runs
from the first part's start to the last part's end, so MetalFX's work and
the handoff are inside every frame time below.

**The inputs**, at the light's size, every one Apple lists (WWDC25
session 211; the SDK headers):

| Input | What it is here |
|---|---|
| color | the frame's light, the image the network reads (checkerboarded, or every pixel traced: section 3) |
| depth | reversed infinite depth as the raster pass's clip has it: near / view depth, near 1 mm, the sky 0 |
| motion | pixels, to where the point was last frame (scale 1), RG16Float |
| diffuse albedo | the surface's color; 0 for conductors (Apple's warning on metals) |
| specular albedo | Karis's EnvBRDFApprox of the surface's F0 and roughness at its viewing angle, Fresnel included; a coated surface's F0 from its index |
| roughness | perceptual (the square root of alpha), linear |
| specular hit distance | enabled: the length of a ray traced along the mirror direction from the surface settled on, 65504 to the sky |
| normal | world space, a signed format (Apple's warning) |
| world-to-view, view-to-clip | the camera's, as the raster pass projects |
| jitter | the sample's offset in pixels, negated (section 2.2) |
| reset history | when the frame's history is not valid |

Auto exposure off: the light is linear radiance at exposure 1; with it on,
the output saturated to white.

### 2.1 Through glass and mirrors: primary surface replacement

NVIDIA's sample for DLSS Ray Reconstruction (`nvpro-samples/vk_denoise_dlssrr`,
`primary_rgen.slang`, `#PSR`), the published integration of the denoiser
MetalFX's is the counterpart of, follows a mirror's reflection to the first
surface that is not a mirror and describes that surface to the denoiser,
as if seen through a window into a mirrored world:

- its albedos and roughness, tinted by the mirrors on the way;
- its normal taken through each mirror's reflection matrix;
- its depth and motion those of a virtual point on the camera's ray, at
  the length of the whole path to the surface; the motion the camera's
  alone (NVIDIA's sample does the same);
- the specular hit distance measured from that surface.

The prototype's first inputs replaced the albedos and roughness this way
(the features pass already follows glass and mirrors for the network,
contract 17) but left depth, motion and normal at the mirror or glass
surface itself, so MetalFX moved reflections with the marble's surface and
rejected its own history there: the glass and mirrors came out speckled.
The final inputs follow NVIDIA's sample whole for mirrors.

Glass is not in NVIDIA's sample. Our extension: at each glass surface the
heavier of the reflected and refracted paths (refraction unless Fresnel's
F is over a half or the ray is totally reflected), as the features pass
already takes every glass surface after the first, and the same virtual
point along the camera's ray. The network's own albedo and normal keep
contract 17's Fresnel blend; MetalFX's inputs are written to images of
their own, so the network, the depth of field and contract 17 are
untouched.

### 2.2 The jitter's sign

The header defines the jitter as "the pixel offset this scaler samples to
return to the frame's reference frame": from the sample back to the
pixel's center. Apple's WWDC25 code passes its projection's jitter with y
negated, which works out to the same: the negative of the sample's offset
in texture space. Ours is the sample's offset, right and down
(`core/sampling/jitter.h`), so both axes are negated. The first round
passed it unnegated.

## 3. Fairness

- **The light.** Our frame traces half the light's pixels each frame and
  fills the other half from the frame before (Milestone 4's checkerboard),
  and our network was trained on that. MetalFX is built for whole frames,
  so each MetalFX size runs twice: on the checkerboarded light, and with
  every pixel traced each frame ("full light").
- **What each frame time includes.** The network's rows are the committed
  build, unchanged. The MetalFX rows include producing its inputs (the
  specular albedo, roughness, hit distance with its traced ray, and the
  replaced depth, motion and normal) at the light's size; the full-size
  features pass, which only feeds the depth of field, does not compute
  them.
- **Sizes.** 1920 x 1080 from a 960 x 540 light (2x each way);
  3456 x 2234 from 1728 x 1117 (2x) and from 1152 x 745 (3x, MetalFX
  only: our network is trained for 2x).

## 4. Results

The scene held at t = 120 s, 800 frames at 1920 x 1080 and 300 at
3456 x 2234 (`--time 120 --times`), each run's mean over its seconds but
the first; three runs each, alternated, the GPU idle before every run;
every configuration's runs within 2% of their mean. The
network's figures match Milestone 6's measured frame (62.6 fps, 15.98 ms;
17.6 fps, 56.8 ms) within 1%.

| Output | Configuration | fps (ms) |
|---|---|---|
| 1920 x 1080 | **our network, 2x** | **63.0 (15.87)** |
| | MetalFX 2x, checkerboarded light | 77.3 (12.93) |
| | MetalFX 2x, full light | 56.6 (17.67) |
| 3456 x 2234 | **our network, 2x** | **17.5 (57.22)** |
| | MetalFX 2x, checkerboarded light | 23.2 (43.18) |
| | MetalFX 2x, full light | 16.2 (61.74) |
| | MetalFX 3x, checkerboarded light | 40.0 (25.00) |
| | MetalFX 3x, full light | 30.0 (33.29) |

The first inputs (section 2.1, 2.2) ran within 2% of these: MetalFX 2x
checkerboarded 78.7 fps (12.70 ms) at 1920 x 1080 and 22.9 (43.67) at
3456 x 2234, 3x 39.5 (25.34).

### Images

Frame 7029, the whole frame above and three crops below, every column the
same frame:

- [1920 x 1080](2026-10-27-metalfx-denoising/metalfx-vs-network-1920x1080.jpg):
  our network; MetalFX checkerboarded; MetalFX full light.
- [3456 x 2234](2026-10-27-metalfx-denoising/metalfx-vs-network-3456x2234.jpg):
  our network; MetalFX 2x checkerboarded and full light; 3x checkerboarded
  and full light.
- [MetalFX's inputs, first and final](2026-10-27-metalfx-denoising/metalfx-inputs-before-after.jpg),
  1920 x 1080: our network; MetalFX on the first inputs; on the replaced
  surfaces and the negated jitter.

What they show:

- **The final inputs fix the speckle.** On the first inputs the striped
  glass marbles and the blue glass were grainy; on the final ones they
  are clean. With the checkerboarded light at 1920 x 1080 a little speckle
  stays on glass rims; full light removes it.
- **Every MetalFX column loses the caustics:** the green glints inside the
  glass marbles, which MNEE traces and our network keeps.
- **Every MetalFX column darkens the mirrors' pinpoints:** the fireflies
  reflected in the mirror marbles' upper halves are bright points in our
  network's frame and a dark grey in MetalFX's.
- Full light and the 3x light change neither.

## 5. Why, as far as can be told

- **Fact:** the lost detail is small, sparse and very bright, on surfaces
  whose albedo, normal and roughness do not predict it.
- **Inference:** a general denoiser treats such pixels as noise and
  smooths them away; our network was trained on this scene's own light
  paths, the caustics and the fireflies' reflections among them.
- **Not tried:** MetalFX's denoise strength mask, which marks areas it
  should not denoise. Apple documents it for areas carrying little
  information (an Apple engineer on the developer forums), not for
  highlights, and marking them would need knowing where they are before
  denoising.
- **Not modelled:** the reflected object's own motion in the virtual
  point's motion (NVIDIA's sample has the camera's alone).

## 6. Left

- The prototype stays outside the repository; nothing of it is committed.
- MetalFX's Metal 4 creation failure is not reported to Apple; the
  standalone program reproduces it in seconds.
- The Neural Engine note's design remains the open route to a faster
  denoiser that keeps the network.

## Sources

- Apple, WWDC25 session 211, "Go further with Metal 4 games" (the denoised
  upscaler's inputs, pitfalls and code):
  <https://developer.apple.com/videos/play/wwdc2025/211/>
- Apple, WWDC26 session 359, "Build real-time neural rendering pipelines
  with Metal": <https://developer.apple.com/videos/play/wwdc2026/359/>
- Apple, the MetalFX SDK headers (`MTLFXTemporalDenoisedScaler.h`,
  `MTL4FXTemporalDenoisedScaler.h`, `MTLFXTemporalScaler.h`), Xcode 26.2.
- Apple Developer Forums, the denoise strength mask:
  <https://developer.apple.com/forums/thread/790642>
- Apple Developer Forums, Metal 4 MetalFX encodes that write nothing:
  <https://developer.apple.com/forums/thread/840795>
- NVIDIA, `vk_denoise_dlssrr`, primary surface replacement:
  <https://github.com/nvpro-samples/vk_denoise_dlssrr>
- NVIDIA, Streamline's DLSS Ray Reconstruction programming guide (the
  guides, the specular albedo approximation, the hit distance's origin):
  <https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md>
- solidean/shaped-core, a MetalFX denoised scaler integration (inverted
  infinite depth, RG16 motion, the jitter's opposite sign):
  <https://github.com/solidean/shaped-core/pull/241>
