# ReSTIR DI Phase A: results, and the 126,002 sample

*2026-10-11. Apple M3 Max, macOS 26.6.2, Xcode 26.2, release build of
`restir-di` at ac6cea8 (`-fmetal-math-mode=fast`). `scenes/marbles.toml`,
1920 x 1080 unless stated. "Naive" is `graphs/path.toml`; "RIS" is
`graphs/restir_ris.toml` (M = 8 candidates, visibility reuse, no merges).*

## Question

At t = 8, frame 65537, pixel (615, 567), the RIS graph produced a sample
of 126,002 (max channel; the reference there is (0.177, 0.089, 0.004)). Over
32 frames RIS showed rare spikes far above anything the naive graph showed.
What produced the sample, and is it a defect or a legitimate rare event?

## Phase A's results

**Frame times.**

| Graph | GPU, 3456 x 2234, t = 120 (c33f186, same shaders as ac6cea8) | Wall clock a sample, 1920 x 1080, t = 8 | Same, t = 120 |
|---|---|---|---|
| naive | 29.40 ms | 8.74 ms | 8.35 ms |
| RIS | 36.17 ms | 10.79 ms | 12.04 ms |

The GPU figures are from 2026-10-10-restir-di-layout.md ("Measured"). The
wall-clock figures are the convergence runs' seconds over their 16,384
samples, with PFM writing included (`make convergence`).

**Error x time** against the 64 x 1024 references, from the main
checkout's `.cache/references/marbles-t*-convergence-16384x1.csv`. MSE x s
is lower-is-better. The last column is naive over RIS, so above 1 means RIS
wins. The references' own MSE floors are 0.133 (t = 8) and 0.0039
(t = 120).

| Samples | t = 8: naive MSE | RIS MSE | naive MSE x s | RIS MSE x s | naive / RIS | t = 120: naive MSE | RIS MSE | naive MSE x s | RIS MSE x s | naive / RIS |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 106.1 | 5.427 | 0.928 | 0.0585 | 15.9 | 1.835 | 2.871 | 0.0153 | 0.0346 | 0.44 |
| 2 | 30.80 | **843.1** | 0.539 | 18.19 | 0.03 | 1.078 | 1.273 | 0.0180 | 0.0307 | 0.59 |
| 4 | 11.61 | **211.7** | 0.406 | 9.135 | 0.04 | 11.62 | 0.924 | 0.388 | 0.0445 | 8.72 |
| 8 | 5.800 | **53.40** | 0.406 | 4.608 | 0.09 | 3.050 | 1.126 | 0.204 | 0.108 | 1.88 |
| 16 | 2.393 | **15.38** | 0.335 | 2.655 | 0.13 | 2.191 | 0.405 | 0.293 | 0.0781 | 3.75 |
| 32 | 1.578 | **4.069** | 0.442 | 1.405 | 0.31 | 1.108 | 0.787 | 0.296 | 0.303 | 0.98 |
| 64 | 0.702 | 1.453 | 0.393 | 1.003 | 0.39 | 0.809 | 0.365 | 0.432 | 0.281 | 1.54 |
| 128 | 0.824 | 0.510 | 0.922 | 0.704 | 1.31 | 1.960 | 0.176 | 2.095 | 0.271 | 7.74 |
| 256 | 0.392 | 0.248 | 0.878 | 0.684 | 1.28 | 2.846 | 0.250 | 6.084 | 0.772 | 7.88 |
| 512 | 0.233 | 0.190 | 1.043 | 1.051 | 0.99 | 0.785 | 0.0825 | 3.355 | 0.509 | 6.60 |
| 1024 | 0.248 | 0.181 | 2.224 | 2.002 | 1.11 | 0.211 | 0.0413 | 1.805 | 0.510 | 3.54 |
| 2048 | 0.205 | 0.366 | 3.670 | 8.096 | 0.45 | 0.0707 | 0.129 | 1.209 | 3.191 | 0.38 |
| 4096 | 0.307 | 0.485 | 10.98 | 21.45 | 0.51 | 0.0335 | 0.0580 | 1.147 | 2.862 | 0.40 |
| 8192 | 0.191 | 0.257 | 13.71 | 22.69 | 0.60 | 0.0592 | 0.0280 | 4.053 | 2.761 | 1.47 |
| 16384 | 0.465 | 0.196 | 66.58 | 34.64 | 1.92 | 0.0211 | 0.0118 | 2.886 | 2.325 | 1.24 |

The MSE rises and falls between rows, which an unbiased estimator's
should not do. Single samples set it. The bold t = 8 RIS rows are the
126,002 sample alone: its error, (126,002, 70,760, 3,561) / n against
the reference, averaged over the image's 3 x 2,073,600 values, gives
840 x (2 / n)^2. That is 840, 210, 52, 13 and 3.3 at n = 2 to 32, against
843, 212, 53, 15 and 4.1 measured. At these frame counts, the tables rank
the estimators by when each one's rarest sample fell, not by how quickly
each one converges.

**Means.** At t = 120 over 1024 samples: RIS 0.18660, naive 0.18611,
reference 0.18637. At t = 8, the mean of 256 single-frame means: RIS
0.25032, naive 0.25011, reference 0.25102. No bias is visible beyond the
noise.

## Tails

Each frame's largest sample (max channel) at t = 8, frames 65536 onward,
one single-frame run per frame, so each image is that frame's own sample.
For RIS, a scratch build that writes only the continuation splits each
sample into two parts. "First surface" is the total minus the
continuation, pixel by pixel: ReSTIR's direct light plus any light the
camera path found before reaching that surface.

| Series | Frames | Median | p90 | Max | Frames over 10k | Over 50k |
|---|---|---|---|---|---|---|
| RIS | 32 | 2,356 | 3,214 | 126,003 | 2 | 1 |
| naive | 32 | 5,744 | 16,513 | 28,544 | 10 | 0 |
| RIS | 256 | 2,354 | 4,516 | 126,003 | 10 | 2 |
| RIS, first surface | 256 | 2,176 | 3,251 | 55,237 | 6 | 1 |
| RIS, continuation | 256 | 972 | 2,953 | 126,003 | 4 | 1 |
| naive | 256 | 4,617 | 15,305 | **165,881** | 58 | 2 |

The 32-frame rows reproduce the table that raised the question. Over 256
frames, the naive graph has spikes as large as RIS's, and as often: a
165,881 at frame 65648 and a 59,058 at frame 65751. The 32-frame window
just happened to contain none of them. Most of the naive graph's 58
frames over 10k fall at one place, around pixel (545, 639). RIS keeps that
place under its median.

## Diagnosis

The 126,002 sample was instrumented. A scratch copy of the shade kernel
recomputed, at that pixel and frame, the record, path state, reservoir,
each candidate (with the initial pass's own numbers), and the
continuation bounce by bounce, with the path's own numbers. It wrote
the values into the image's first row. Its replay of the continuation
equals the shaded value to the bit. The scratch code was not committed.

**The term:** the continuation, not ReSTIR's direct light.

| | Value |
|---|---|
| Pixel's sample | (126,002.6, 70,760.3, 3,561.2) |
| ReSTIR's direct light, x throughput | (5.3e-4, 2.1e-4, 3.6e-6) |
| Continuation | (126,002.6, 70,760.3, 3,561.2) |

**The first surface q:** the camera's first hit (surface 0, throughput 1),
`yellow_porcelain`: coated, albedo (0.88, 0.70, 0.08), coat IOR 1.5, at
(-0.0582, 0.7605, -0.0447).

**ReSTIR's direct light at q, which is sound.** The reservoir chose light
447 (candidate 2) with M = 8, w_sum = 2.11e-3, p̂ = 5.32e-8 and
W = 4,958.6.

- The chosen light is 0.357 m away. Its cos at q is 0.135, and its cos
  at the light is 0.04876 at the decoded point against 0.04880 at the
  continuous one.
- f |cos θ_x| L_e = (0.123, 0.0496, 0.00085), and G = 8.63e-7.
- Of the 8 candidates, only candidate 2 has a nonzero weight. Lights 551,
  311, 2, 82, 67, 534 and 392 are either behind q's shading normal or
  dark at t = 8 (their luminance of f |cos| L_e is 0).
- The shaded value is (f L_e / lum) x w_sum / M. It is finite and small.

**The continuation.**

1. At q, the base's diffuse lobe was sampled (pdf 0.0627), giving
   throughput (0.551, 0.358, 0.025). The ray hit primitive 29 at
   7.5 cm: a steel marble, conductor, f0 (0.56, 0.57, 0.58), GGX α =
   10^-3 (roughness 0.03, clamped at `least_alpha`).
2. Next-event estimation (NEE) at the steel chose light 587, with
   probability 1/616. It drew a direction 4.8 cm to the light, with
   L_e = (30.6, 26.0, 18.4) and a cone density of 347.6 / sr (a cone of
   2.9e-3 sr). That direction lands in the GGX lobe: f = (4,281, 4,358,
   4,434), cos θ_i = 0.984. The red channel is

       0.551 x 4,281 x 0.984 x 30.6 / (1/616 x 347.6) = 126,000

   That is this sample.
3. The next BSDF sample at the steel (GGX, pdf 51,609) hit light 638 at
   4.8 cm. It contributed nothing, because the path tracer counts emission
   only after a delta lobe (`path.metal.h`, step 6).

**The checks asked for:**

| Check | Finding |
|---|---|
| Candidate's weight and shaded value use the same point and density? | Yes. Both are evaluated at the decoded point (`draw_candidate()`, `unshadowed()`). The decoded and continuous points agree to within the arithmetic's rounding: cos at the light differs by under 3 x 10^-5, and the angle between the two directions is 0 or 3.5 x 10^-4 rad, which is the resolution of acos near 1 in float. |
| Quantization at the rim of a near light's cap | Not involved. The survivor's cos at the light is 0.049, not near 0, and no candidate's sign flips on decoding. |
| Packed BSDF tolerance on a sharp coat lobe | Not involved. The coat lobe is delta and is never evaluated by a candidate. The record's packed values are within half's tolerance: α 1.0004e-3 against 10^-3 on the steel at frame 65594, and 3.5992e-3 against 3.6e-3 on the copper. |
| f and p̂ disagreeing for colour | Yes, as designed. p̂ is luminance, so the shade's max channel is w_sum / M times max(fL) / lum(fL). Here that factor is 1.99, and at the two direct spikes below 1.47 and 1.14 (its bound is 1 / 0.0722 = 13.9). |
| A near-zero p̂ dividing w_sum | Here p̂ = 5.3e-8 and W = 4,959, but the survivor is the only weight in w_sum, so the two cancel. A small-p̂ survivor beside a large w_sum happens with probability w_y / w_sum, and pays exactly that. |

**ReSTIR's own direct spikes (6 frames of 256 over 10k) have the same
cause, at the first surface.** Two were instrumented:

| Frame, pixel | Sample | First surface | Survivor | Its weight, of w_sum | W | Naive NEE at the same draw |
|---|---|---|---|---|---|---|
| 65548, (1338, 700) | 42,216 | copper, α 3.6e-3 | light 569 at 6.3 cm; f = 33,315 (R) | 229,855 of 229,855 | 366 | about 338,000 |
| 65594, (1055, 517) | 55,237 | steel, α 1.0e-3 | light 523 at 11.2 cm; f = 64,447 (R) | 386,995 of 386,995 | 250 | about 441,000 |

Each time, one light candidate out of 8 landed in a near-mirror lobe. RIS
returns that candidate's weight divided by M, which is one eighth of what
naive NEE returns for the same draw. The naive graph's 165,881 lies on a
brass first surface (α 2.5e-3). That pixel's first surface was read from
the RIS record, which the same camera numbers produce. Its NEE was not
instrumented, so this cause is an inference.

## Verdict: legitimate; not a ReSTIR defect

The sample is the naive path tracer's NEE, with no multiple importance
sampling (MIS), at a near-specular conductor. The RIS graph shares that
estimator from q's first bounce onward, and the naive graph uses it
everywhere.

Light sampling draws a direction uniformly over the light's cone Ω,
choosing among N lights. The estimate f cos L_e N Ω therefore ranges
from 0 to N Ω f_peak L_e. For GGX, f_peak is about F / (4π α² cos θ_o).
At α = 10^-3, with N = 616 and fireflies a few centimetres from the
marbles, the peak reaches 10^5 to 10^6. It is hit with probability
roughly (the lobe's solid angle, ~α²) / Ω per draw.

- The estimator is unbiased. The means agree with the reference.
- Its variance is finite but grows as 1/α², so the distribution is heavy
  tailed: rare, huge samples.
- RIS over M light candidates divides each such spike by M and makes it
  M times as frequent. That is why the first-surface tail is lighter in
  RIS than in the naive graph.
- No light-only resampling can remove these spikes. The density that
  matches a near-mirror lobe is the BSDF's.

The defect, if the word applies, is in the shared light transport: no MIS
between light sampling and BSDF sampling on glossy lobes. The BSDF sample
that found light 638 was thrown away, when MIS would have weighted it
instead. That is not ReSTIR's, so nothing was changed here.

## Production

- **MIS of light and BSDF sampling** (Veach 1997, chapter 9; pbrt-v4's
  `PathIntegrator`) is how path tracers handle this. NEE weights its
  sample by the power heuristic, and a BSDF-sampled ray that hits a light
  off a non-delta lobe counts that light with the complementary weight.
  For the 126,002 draw, the light density (selection x cone) is 0.564 / sr
  and the GGX density is 7,524 / sr. The power heuristic weighs the NEE
  sample by 5.6 x 10^-9. The BSDF path carries that light at f cos /
  pdf ≈ 0.56 x L_e.
- **RTXDI's initial sampling** draws BRDF-sampled candidates alongside the
  light candidates: by default 8 local lights and 1 BRDF sample, with a
  BRDF cutoff of 10^-4. Every candidate's source density is the blend
  (n_light p_light + n_brdf p_brdf) / (n_light + n_brdf) in solid angle,
  a balance heuristic across the two strategies. RTXDI skips the BRDF term
  only for analytic light samples, which BRDF rays cannot hit. Serenity's fireflies are
  shapes that rays can hit.
- **RTXDI's boiling filter** (on by default, strength 0.2) runs after
  temporal resampling. It empties reservoirs whose w_sum is far above
  their thread group's. It targets feedback in temporal reuse. The RIS
  graph has no temporal pass, and the filter never touches the
  continuation, so it does not apply here. It belongs with the temporal
  merge.
- **Clamping** each sample's radiance removes the spikes but biases the
  mean. The RTXDI sources read here contain no such clamp, so no claim is
  made about whether production uses one.

## Recommendation (not implemented)

1. **MIS in the path tracer.** Weight NEE and BSDF sampling by the power
   heuristic at every surface that aims at lights, counting emission on
   BSDF-sampled hits off non-delta lobes. This removes the continuation's
   spikes and the naive graph's own. It changes the path pass's images, on
   purpose, though not their expectation. That breaks the "bit-identical
   to main" invariant, so it is a change of its own: declared, measured
   against the references, and with the 48-PFM check replaced by a
   statistical one.
2. **A BSDF candidate in ReSTIR's initial sampling,** as RTXDI ships it:
   1 BSDF candidate beside the 8 light candidates. Each candidate's
   density is the blend above, converted to the target's area measure
   (`target.metal.h`). Its continuation then counts a light hit by the
   BSDF ray at q with the matching MIS weight. This removes ReSTIR's
   remaining first-surface spikes at sharp metals.
3. **The boiling filter with the temporal merge,** at RTXDI's defaults.
4. **No clamp** in any image measured against the reference.
5. **Measure error x time over enough frames to be robust to the tail,**
   for example as the median of independent batches, as the reference's
   floor already is. At up to a few thousand samples, the tables above are
   ordered by their rarest samples.

## Method

- **Instrumentation.** A scratch copy of
  `src/metal/passes/restir_shade/restir_shade.metal`, built into a
  scratch binary and restored from git afterwards. At one pixel and frame,
  it replayed steps 1 to 3 and the continuation, and wrote 120 float3s
  into row 0. The pixels of row 0 skip their own write in that frame.
- **Tail split.** A second scratch build accumulates only the
  continuation: the shade's radiance minus (the path state's light plus
  throughput x direct).
- **Single-frame runs.** `serenity-headless --graph G --scene
  scenes/marbles.toml --size 1920x1080 --first F --frames 1 --time 8
  --format pfm --write all`, for F = 65536 to 65791. The convergence runs
  use the same frame indices.
