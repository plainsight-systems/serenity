# The reference, and the error against it

*2026-10-10. Design note for Milestone 1's last item; the headers are the
design (core/contracts/linear_image.h, contract 13; core/output/pfm.h,
image_format.h; core/measurement/reference.h, error.h; headless/options.h
--format; measure/options.h). This note keeps the facts behind it and what was set
aside.*

## Question

Every estimator after the naive one (ReSTIR DI, GI, the denoiser) is to be
judged against a reference (logical-overview.md): the same light transport,
time frozen, many samples. The architecture already says the reference is
settings, not code (change-axes.md), and the headless renderer can render
it (--time, --samples, --write last). What it cannot do is keep it, or say
how far an image is from it, or how far the reference is from the truth.

## Facts

- The headless renderer writes PNGs: 8 bits a channel, tone mapped. A
  reference must keep linear radiance at float precision; the GPU's
  accumulated image has it (metal/frame/renderer.h, read_accumulated), and
  nothing writes it.
- That image is a running mean in float (metal/film/accumulate.metal.h):
  mean' = mean + (sample - mean) / (n + 1). A step under half the mean's
  last place, 2^-24 of it, is lost: at n = 2^16, any sample within
  2^16 x 2^-24 = 0.4% of the mean changes nothing. At n = 2^10 the band is
  0.006%.
- pbrt-v4 (src/pbrt/util/image.cpp, cmd/imgtool.cpp) measures MAE, MSE,
  MRSE = mean of (v - r)^2 / (r + 0.01)^2, and FLIP, summing in double;
  imgtool's MRSE skips a pixel whose relative error is infinite. It reads
  and writes PFM and EXR.
- Falcor's ErrorMeasurePass loads its reference from EXR or PFM, measures
  L1 or L2 (MSE) per channel, and writes a CSV row a frame.
- The ReSTIR GI paper (Ouyang et al. 2021) reports its gains as MSE.
  Whether the ReSTIR DI paper (Bitterli et al. 2020) reports MSE or a
  relative MSE was not checked: its PDF is 50 MB, and the search found
  only its abstract.
- At 1920 x 1080 the path graph's sample costs some 8 ms (the 3456 x 2234
  baseline, 29.9 ms, scaled by pixels): 65,536 samples is some 9 minutes;
  at 3456 x 2234, some 33 minutes.

## Design, and what was set aside

| Choice | Kept | Set aside, and why |
|---|---|---|
| The file | PFM: a ten-line format, read by pbrt-v4, Falcor and tev, written exactly | OpenEXR, by tinyexr or the OpenEXR library: a dependency, and compression and channels nothing here needs |
| Many samples | Batches: B images of N = 1024 samples, independent by frame index, combined in double | One image of B x N samples: the float running mean's lost steps (above); the GPU in double: Metal has no double |
| Combining | Welford's (n, mean, M2) a channel, in double, the batches' order fixed (CDSA.23) | Sums of values and of squares: cancels where the variance is small beside the mean |
| The reference's own error | Its floor, from the batches' variance: var / B a value, averaged | None stated: an error near the reference's own would read as the estimator's |
| Checking the floor | Once: two references from disjoint frames, whose error against each other should be some twice the floor | Every reference made twice: doubles every reference's cost to check a formula |
| Measures | MSE and relative MSE (pbrt's MRSE, its epsilon) | FLIP: perceptual, and wanted for the denoiser, but NVIDIA's implementation is a dependency of its own; added with the denoiser. MAE: neither pbrt's default nor Falcor's L2 |
| Bad values | Refused, naming the pixel | Skipped, as pbrt's MRSE skips infinities: an error over fewer pixels understates it silently |
| Size | 1920 x 1080 for error, the size the image judged is rendered at | 3456 x 2234: four times the cost; the display's size is kept for time (the baseline), not for error |
| Times | t = 120, the time baseline's (every firefly awake), and t = 8, the opening (few awake) | t = 0: no light at all; error against black is 0 for any estimator |

The headless renderer gains --format pfm, the accumulated image's radiance
before tone mapping; a new program, serenity-measure, makes a reference
from batches and measures images against it, on the CPU. No shader or pass
is added, and no frame of the window costs more. A reference does cost the
GPU what its graph does: graphs/path.toml is the path pass and then the
tone map, so every sample of a batch also tone maps, into an image a PFM
never reads; and each batch's PFM is one readback of the accumulated
image, 16 bytes a pixel, 33 MB at 1920 x 1080, copied once (GPU.1). Both
are budgeted below, under the review.

## How it is run (the Makefile's targets)

- `reference`: SCENE through graphs/path.toml at TIME and SIZE, batch k
  rendered as `--first k --frames 1 --samples N --time TIME --format pfm`,
  k = 0 .. B - 1, so batch k's samples are frame indices k N .. k N + N -
  1, none shared; then `serenity-measure reference`. Defaults B = 64, N =
  1024: 65,536 samples a pixel.
- `convergence`: the naive path tracer at TIME from frame index F on, F N'
  at least B N so none of its samples is the reference's, M samples a
  frame, written at doubling frames (--write doubling --format pfm), each
  measured by `serenity-measure error`: a CSV of samples, MSE and relative
  MSE.

## What it is checked by

- Unit tests on the CPU: PFM round trip and every refusal; Welford's mean
  and variance against two-pass sums in long double on small images; the
  floors on images of known variance; both measures on images whose errors
  are known in closed form.
- The two references checked against each other, once (above).
- The naive estimator's convergence: its MSE against the reference falls
  as 1 / samples, a slope of -1 on doubling axes, until it nears the floor.
  A slope that is not -1 says the reference, the measure or the estimator
  is wrong.

## Review

Codex reviewed the design (4376269, both focuses). Taken, in the headers:

- A PFM's scale is a factor on every value, which pbrt-v4's reader
  applies; the reader here took any negative scale and ignored it. Now a
  scale other than exactly -1 is refused.
- The headless renderer accepted sample indices past 2^32, while the
  shaders key random numbers by the low 32 bits: a batch from --first
  4194304 at 1024 samples would have drawn batch 0's numbers again,
  silently. Indices past 2^32 - 1 are now refused (headless/options.h).
  Older than this design; found by it.
- The image passed between Film, Output and Measurement is now contract
  13 (core/contracts/linear_image.h), not a Film header the others
  include (file-mapping.md: families depend on contracts).
- Its side limit no longer derives from Metal's: the core's own bound on
  what a file may make a reader allocate.
- The list of output formats is Output's (core/output/image_format.h), not
  the headless renderer's.
- This note said the reference costs the GPU nothing beyond its frames;
  corrected above.

Set aside, as costs too small to design around (Per.2), each to be
confirmed by timing a batch when the first reference is rendered:

- Frozen time rebuilds the same acceleration structure every sample:
  animate() is 43 µs a frame and a rebuild some 95 to 161 µs, so some 9
  to 13 s of a 65,536-sample reference's some 9 minutes, 2%. Caching the
  scene by its time would touch the renderer's frame loop for every run.
- The tone map runs every sample into an image no PFM reads: at 1920 x
  1080 some 0.2 ms a sample (0.6 ms over the display pass at 3456 x 2234,
  2026-10-09-pass-costs.md, scaled by pixels, the display pass's own
  share unmeasured), some 13 s of a reference, 2%. A graph that ends
  after the path pass would change the rule that a graph computing light
  ends in a presenting pass (core/frame/schedule.h), for every graph.

## Results

To come: the references at t = 120 and t = 8, their floors, the two-
reference check, and the naive estimator's convergence at both times.
