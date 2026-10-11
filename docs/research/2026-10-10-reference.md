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
image, copied three times on its way (GPU.1): GPU to a shared buffer, 16
bytes a pixel; that buffer into the caller's span, 16 again
(read_accumulated); and the span's radiance into a new image, 12
(from_rgba). That is 33, 33 and 25 MB at 1920 x 1080, once a batch, beside
some 8 s of rendering a batch. Both costs are budgeted below, under the
review; the copies, a fraction of a second a reference, are not designed
around.

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

## Implementation

Built on the design above; the guideline pass over it is
[2026-10-10-reference/guidelines-pass.md](2026-10-10-reference/guidelines-pass.md).
The headers changed in the implementation, each for its reason:

- core/contracts/linear_image.h gains `checked_values()`, the invariant's
  check every reader makes, and `value_position()`, "pixel (x, y) channel
  c": four readers had each written the check, and two the naming (ES.3).
- core/measurement/error.h: `error_against()` takes a `Judged {image,
  reference}`, the two images by name; side by side, swapped arguments
  gave other numbers silently, the measure being asymmetric (I.24).
- core/output/pfm.h's cost note: a file is read whole with one stream
  operation, its rows then turned in place, and written a row at a time,
  bottom first, through the stream's buffer; "written whole with one
  stream operation" would have needed a turned copy of the image.
- headless/options.h: its first paragraph named the files .png only; it
  now says .pfm with --format pfm, and that PNG is the default.
- measure/options.h: the error's CSV quotes a path holding a comma, a
  double quote or a line break as RFC 4180 has it, so no path splits its
  row.

The Makefile's targets write `.cache/references/<scene>-t<TIME>-<SIZE>-<B>x<N>.pfm`,
its floors beside it as `.txt`, and the convergence's CSV beside it as
`<same>-convergence-<frames>x<samples>.csv` (columns samples, image, mse,
relative_mse). A reference is never replaced: `make reference` stops
before rendering if its file is there.

### The review of 5f4c7de

Codex reviewed the implementation (both focuses). Taken, with the headers
changed for each:

- The sample-index bound was checked by parse() at --samples, while a
  graph that accumulates nothing renders one sample a frame: a valid run
  was refused. parse() now bounds the last frame index alone; the samples
  bound is `check_samples()`, made at the plan's samples a frame once the
  graph is read, before anything renders or the directory is made
  (headless/options.h).
- "A reference is never replaced" was a check, then a truncating open: two
  runs could both pass it. `write_pfm()` now creates its file by exclusive
  create, `std::fopen`'s "x", which is the check, and removes its own
  partial file if a write fails (core/output/pfm.h); serenity-measure has
  no check of its own (measure/options.h).
- The headless renderer named Output's kinds. Output now says, for each
  kind, its names, extension and source (displayed or accumulated), and
  writes any kind by `write_image()` (core/output/image_format.h); the
  headless renderer reads back by source and names no kind
  (headless/options.h).
- Three cost statements were wrong: the Makefile's "most of an hour" (some
  ten minutes, above); "copied once" (three copies, above); error.h's "one
  pass" (two, its check and its sums, so a refusal precedes any sum;
  reference.h states its two passes a batch the same way).

Rejected again, as at the design's review (above): caching the frozen
scene's acceleration structure by time, and a graph without the tone map,
each some 2% of a reference; their measured share comes with the first
references.

## Results

Rendered at 4cb5070's code (5f4c7de for the renders; the later commit
changed no pixel), release build, M3 Max, `make reference` and `make
convergence` with their defaults: 1920 x 1080, 64 batches of 1024 samples,
65,536 a pixel; convergence 16,384 frames of one sample, from frame 65,536
on, written at doubling frames.

| | t = 120, every firefly awake | t = 8, the opening |
|---|---|---|
| Reference, wall time | 9 min 27 s | 9 min 39 s |
| MSE floor | 0.00386 | 0.133 |
| Relative MSE floor | 0.00173 | 0.00213 |
| Convergence, wall time | 2 min 33 s | 2 min 51 s |

The convergence curves, error against samples:

| Samples | MSE, t = 120 | relative MSE, t = 120 | MSE, t = 8 | relative MSE, t = 8 |
|---|---|---|---|---|
| 1 | 1.84 | 37.1 | 106 | 174 |
| 2 | 1.08 | 27.4 | 30.8 | 187 |
| 4 | 11.6 | 55.1 | 11.6 | 66.6 |
| 8 | 3.05 | 16.6 | 5.80 | 21.1 |
| 16 | 2.19 | 997 | 2.39 | 8.28 |
| 32 | 1.11 | 251 | 1.58 | 4.49 |
| 64 | 0.809 | 76.8 | 0.702 | 2.18 |
| 128 | 1.96 | 25.4 | 0.824 | 302 |
| 256 | 2.85 | 21.7 | 0.392 | 76.8 |
| 512 | 0.785 | 7.13 | 0.233 | 19.9 |
| 1024 | 0.211 | 3.19 | 0.248 | 9.71 |
| 2048 | 0.0707 | 1.71 | 0.205 | 6.10 |
| 4096 | 0.0335 | 0.763 | 0.307 | 37.1 |
| 8192 | 0.0592 | 1.27 | 0.191 | 12.4 |
| 16384 | 0.0211 | 0.482 | 0.465 | 9.17 |

The two-reference check at t = 120: a second reference from batches 64 to
127, disjoint from the first's, against the first: MSE 0.0143 where twice
the floor is 0.0077; relative MSE 0.220 where twice the floor is 0.0035.

Facts, from these:

- The curves do not fall as one slope. One image's error jumps by ten
  times between doublings and back.
- 94.7% of the MSE between the two references comes from 0.01% of the
  pixels, some 200 of 2.07 million, and 92.6% of the relative MSE.
  Without them the MSE is 0.00076. The worst pixel is 0.07 in one
  reference and 49.8 in the other: at 65,536 samples, a pixel whose mean
  rare samples of the order of 10^6 still set. They lie among the marbles
  (rows 334 to 722, columns 386 to 1370 of 1080 x 1920).
- The two references' means agree: 0.18651 and 0.18637, within 0.08%,
  and within 0.00% without those pixels. No bias is seen.
- Where the t = 120 curve is past 1024 samples, the error is what the
  floor predicts for an unbiased estimator: some 65,536 / N floors of its
  own plus the reference's (at 16,384, 0.019 predicted, 0.021 measured).
  Below that it is far under the prediction: the rare samples that set the
  expected error have not yet happened in most images.
- The batches' floor underestimates the reference's error: twice the
  floor is 0.0077 and the two references differ by 0.0143 in MSE; in
  relative MSE, 0.0035 against 0.220, sixty times, since a rare sample in
  the image judged, over a near-black reference pixel, is divided by that
  pixel's (r + 0.01)^2.

Inference: the naive estimator's samples are heavy-tailed. A sample's
value ranges over orders of magnitude, its rare largest ones set its
expected error, and a single image's error is a poor estimate of that
expectation: the measure is right, and one measurement of it is not
enough. Which paths carry those samples is not yet known; the likely ones
are next event estimation's f |cos| L / (P p) on the glossy coat and the
metal, P = 1/616, toward a light close by.

## Decision

Open, for the user: how the error is to be estimated so that estimators
can be compared on it (see the reply that follows this note's commit).
