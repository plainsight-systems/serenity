# ReSTIR DI: the header layout

*2026-10-10. The plan for Milestone 2's headers: which files, what piece of
the algorithm each owns, and the guidelines behind each. The headers, once
written and reviewed, are the design; this note is the map they are written
from, and keeps the facts and the choices that should not live in them.*

## The algorithm, in the order a frame runs it

ReSTIR DI (Bitterli et al. 2020), applied where logical-overview.md applies
it: at the first rough surface the camera's path reaches. Every later bounce
keeps the naive estimator's next event estimation.

1. **Surfaces.** The camera's path, as the path tracer runs it, up to the
   first surface with a lobe that is not delta. Its result is the pixel's
   surface record: where that surface is, how it scatters (its resolved
   BSDF), the direction it is seen from, and the path's state there
   (throughput through the glass and metal before it, medium, the light
   found on the way).
2. **Candidates.** At that surface, M lights chosen as the naive estimator
   chooses one (uniformly), each with a point on it; each weighed by
   p̂ / p, p̂ the target function (the light it would deliver, ignoring
   shadows) and p the density it was drawn with. One survives, in a
   reservoir, with probability its weight over the sum: resampled
   importance sampling (Talbot et al. 2005).
3. **Visibility reuse.** One shadow ray toward the survivor; if it is
   blocked, the reservoir's weight W becomes 0 (its count M is kept). What
   a pixel shares with its neighbors and its next frame is then a light it
   can see: the paper's main quality gain in an occluded scene, and this
   scene is marbles.
4. **Temporal merge.** The reservoir this surface held last frame, found
   where the surface was then, merged in, its count capped at H x M.
5. **Spatial merge.** The reservoirs of k neighbors within a radius,
   merged in.
6. **Shade.** One shadow ray toward the survivor; the direct light is
   f |cos| L_e G V x W, W the reservoir's contribution weight. The path
   then continues from the surface as the naive path tracer does, for the
   indirect light, and the pixel's radiance is folded into the accumulated
   image.

Merges 4 and 5 normalize as RTXDI does, by MIS-like weights over the
inputs (the balance heuristic). Each input reservoir r_i, from surface q_i
with count M_i (the temporal input's capped at H times this pixel's), is
streamed with weight p̂_q(y_i) W_i M_i, p̂_q this pixel's target; with y
kept from input s, W = w_sum pi / (p̂_q(y) piSum), where pi = d_s(y) and
piSum = sum_i M_i d_i(y), d_i input i's domain target at y as its own
frame lit it. The graph's `bias_correction` says what d_i holds (the
production rework, below):

- **basic** (the default, RTXDI's `Basic`): d_i = p̂_{q_i}, unshadowed. No
  ray in a merge. Visibility reuse put visibility into what every
  reservoir holds and these weights leave it out, so an input whose
  surface cannot see y still takes a share of it: a small darkening on the
  lit side of shadow edges, which production accepts. The shade traces the
  survivor's shadow ray, or reuses a test of it made within 4 frames and
  16 pixels (RTXDI's `reuseFinalVisibility`).
- **raytraced** (unbiased, for checking against the reference): d_i =
  p̂_{q_i} V_{q_i}, one shadow ray from each input's surface but the
  source's (visible by the merges' invariant); the temporal input's
  tested as last frame saw it, its lights where they were, against last
  frame's acceleration structure; and the pixel's own domain tested by the
  ray that also decides the survivor's visibility, so the shade traces
  none. The same weights by the same code as basic, so this mode checks
  what basic ships.

Rays a pixel, against the naive estimator's one at that surface: basic, 1
for visibility reuse and at most 1 to shade, 1 to 2; raytraced, 1 for
visibility reuse, 1 for the temporal merge's test, k for the spatial
merge's: k + 2, 3 at RTXDI's k = 1.

How production does it: NVIDIA's RTXDI library splits ReSTIR DI the same
way (initial sampling, temporal, spatial, pairwise MIS, reservoir storage),
stores a light by index and a point on it (`lightData`, `uvData`), and
defaults to 8 light candidates, a history of 20, a spatial radius of 32
pixels, a depth threshold of 10% and a normal threshold of cos 0.5. Its
library is under NVIDIA's proprietary license: read for its design, none
of it copied. Since the production rework (below), its shipped defaults
are Serenity's.

## Additive: the naive path tracer stays as it is

Every change adds beside what exists. The naive path tracer's graphs read
as now, render the same images to the bit, allocate the same images and
cost the same time; ReSTIR's images, constants and previous-frame copies
exist only for a graph that has its passes. The one edit to existing code
is path.metal.h's split into two functions the path pass calls in turn.
Held by: the existing GPU tests unchanged; the path pass's accumulated
image compared bit for bit before and after on every scene; and the path
graph's frame time measured again against its 29.90 ms baseline.

## The parallel with attention

| ReSTIR DI | Attention |
|---|---|
| Target p̂(y) at a surface | The score q · k for a key |
| Candidate weight p̂ / p | exp(score), unnormalized |
| Reservoir (y, w_sum, M) | Online softmax state (m, l), one value kept |
| Picking y with probability w / w_sum | Sampling a key from softmax(scores) |
| Merging two reservoirs | FlashAttention combining two blocks' (m, l) |
| Temporal reuse, history capped | A KV cache with a window |
| Spatial reuse over k neighbors | Local attention over a neighborhood |
| Shading the survivor with a shadow ray | Reading the chosen value |

The difference worth stating: a reservoir keeps one sample and a weight;
attention keeps a weighted sum of values. Picking by w / w_sum over a
stream is the same draw as GDSA.21's random-key race (Gumbel-max is an
exact draw from softmax), so a reservoir fed exp(score) picks key i with
probability softmax(score)_i. That is a GPU test (docs/process/MEMORY.md),
and the reservoir header states it.

## The headers

Twelve files new or added to; every change is additive (above). Core first, then the Metal backend. Each entry:
the piece of the algorithm it owns, then the guidelines that shape it.
The sizes, settings and passes in these entries are this note's first
plan; where the headers depart from it, the table at the end says so, and
its rows marked (rework) give what the headers say since the production
rework.

### Core (platform-neutral)

**`core/passes/restir_di.h`** (new; Pass axis, as `core/passes/tone_map.h`)
- The estimator's settings, data not code (change-axes.md): candidates M
  (default 8), history cap H in units of M (20), neighbors k (5), radius in
  pixels (30, Bitterli's), the depth and normal thresholds a neighbor must
  pass (0.1, 0.5), each with its range.
- Guidelines: C.2 (plain settings, a struct); I.5 and E.2 (ranges checked
  where read, refused naming the line); ES.45 (every default named);
  Per.7 (the settings are what a measurement varies, so they are data).

**`core/frame/schedule.h`** (changed)
- Five pass kinds: `surfaces`, `restir_candidates`, `restir_temporal`,
  `restir_spatial`, `restir_shade`. What each reads and writes of the
  frame's images, and the order a schedule must keep (candidates after
  surfaces, temporal and spatial after candidates, shade after the last
  merge and before a presenting pass).
- An estimator is a graph: `["surfaces", "restir_candidates",
  "restir_shade"]` is RIS alone; with M = 1 it is naive next event
  estimation at the first surface (principle 4: estimators are
  interchangeable). The reference is the path pass, unchanged.
- Guidelines: Enum.3 (an enum class); P.6 (a switch with no default, so a
  backend missing a kind fails its build, as now); I.5 (a schedule's order
  checked when it is read).

**`core/frame/graph_file.h`** (changed)
- Reads `[restir_di]` beside `[tone_map]`: present exactly when the graph
  has a ReSTIR pass.
- Guidelines: SL.io.2 (ill-formed input refused), E.14.

**`core/frame/history.h`** (changed)
- Reservoirs and surface records are history, as the accumulated image is:
  a frame with temporal reuse is a function of the frames since its history
  started. What starts it over: a new size, a new graph, a camera cut. What
  does not: moving fireflies, which the merge re-evaluates (principle 6).
  The headless plan replays the frames since the history started, so frame
  k rendered alone is frame k rendered in a run (principle 2, as for
  accumulation now).
- Guidelines: I.7 (the postcondition: the same frames, the same image).

**`core/contracts/light_sample.h`** (new; contract 4, owned by Light)
- A light sample by identity: the light's index and a point on it in its
  own coordinates; for a sphere, the unit direction from its center,
  octahedral-encoded in two 16-bit numbers (RTXDI's `uvData`). Evaluated
  again wherever and whenever it is reused, so a moving firefly's sample
  follows it (principle 6). 8 bytes.
- Guidelines: shared-layout rules as every contract (static_assert on size,
  trivially copyable, P.5); ES.46 (the 16-bit quantization stated, and its
  error: 2^-16 of the sphere's extent, under a firefly's 1.5 mm by far).

**`core/contracts/surface_record.h`** (new; contract 5, "Film outputs",
owned by Film)
- What a pixel's first rough surface is, for the merges and the shade:
  position (12 bytes), the direction it is seen from (octahedral, 4), its
  resolved BSDF (contract 2, 48): 64 bytes, read by every merge. The path's
  state there (throughput, medium, eta scale, the light found on the way,
  the path's numbers' counter) in a second record read only by the shade,
  so the merges' neighbor reads carry only what the target needs.
- Guidelines: GPU.2 (laid out for coalesced access: one record a pixel,
  a flat array in pixel order); GDSA.6 (its size counted in passes over
  memory, below); CDSA.31 (quantize what tolerates it: the two directions;
  position and BSDF stay full precision); CACHE.6 (hot fields, what the
  merges read, split from cold ones, the path state).

**`core/contracts/reservoir.h`** (new; contract 14, owned by Sample reuse;
read by Pass and Frame graph)
- A reservoir as stored between passes and frames: the light sample
  (contract 4, 8 bytes), the contribution weight W (4), the count M (4):
  16 bytes, aligned so a SIMD group's loads coalesce.
- Guidelines: GPU.2; CDSA.23 (a small, mergeable state: the whole
  streaming state of resampling is these 16 bytes); P.5.

**`core/contracts/reuse_constants.h`** (new; owned by Sample reuse,
read by the ReSTIR passes)
- The previous frame's camera and time, for the temporal merge's
  reprojection and its targets evaluated as the previous frame lit them;
  the settings the shaders read (M, H, k, radius, thresholds). Its own
  buffer, bound only by the ReSTIR passes, so contract 7, which every pass
  binds, is unchanged.
- Guidelines: shared-layout rules (P.5); I.5 (settings checked on the CPU
  before they are written).

### Metal backend

**`metal/reuse/reservoir.metal.h`** (new; Sample reuse)
- The reservoir in registers and its operations: stream a candidate
  (w = p̂ / p, keep it with probability w / w_sum); stream another
  reservoir (w = p̂_q(y_i) W_i M_i); and finish, W = w_sum / (Z p̂_q(y)),
  Z as the caller counted it (1 for candidates alone, so W = w_sum / (M
  p̂) as RIS gives). The one merge rule every reuse uses, direct and later
  indirect (change-axes.md).
- The softmax parallel, and the test that holds it (above).
- Guidelines: CDSA.23 (update and merge, a fixed order); GDSA.21 (the same
  draw as the random-key race); GDSA.3 (each choice's number from pixel,
  frame and purpose, metal/sampler).

**`metal/reuse/unbiased.metal.h`** (new; Sample reuse)
- Z, the paper's unbiased normalization: for the survivor y, the sum of
  the counts M_i of the inputs whose surface supports y, p̂_{q_i}(y) > 0
  and y visible from q_i, the visibility asked of a function the caller
  passes (the integrator's shadow ray, against this frame's structure or
  last frame's). Why it makes the merge unbiased, stated: every sample the
  merge could have chosen is counted by the domains that could have made
  it (Bitterli et al. 2020, section 4.3).
- Guidelines: I.7 (the postcondition, E[W f] = ∫ f, held by the GPU test
  that ReSTIR converges to the reference); P.1.

**`metal/reuse/temporal.metal.h`, `metal/reuse/spatial.metal.h`** (new;
Sample reuse)
- Temporal: where this surface was last frame (its position through the
  previous camera), whether that pixel's surface is the same one (depth
  within 10%, normals within cos 0.5), the history cap H x M, and Z's test
  of the previous surface, as last frame lit and occluded it. A surface
  seen through glass reprojects wrongly; the similarity test and Z make
  that noise, never bias (logical-overview.md).
- Spatial: k neighbors drawn within the radius, each tested as above, and
  Z's k visibility tests.
- Guidelines: GDSA.3; SIMD.5 and GPU.2 (neighbor reads are gathers within
  a 30-pixel disc: cached, not coalesced; counted below, measured).

**`metal/integrator/target.metal.h`** (new; Integrator)
- The target function p̂ at a surface for a light sample: the luminance of
  f(wo, wi) |cos θ_x| L_e cos θ_y / d², in area measure on the light, so
  one sample means the same thing at every surface it is reused at. The
  conversion from the sphere light's solid-angle density to area measure.
  The Integrator owns it: it defines what "contribution" means
  (change-axes.md).
- Guidelines: ES.3 (one definition, used by candidates, merges and shade);
  F.4 (pure, inline).

**`metal/integrator/restir_di.metal.h`** (new; Integrator)
- The estimator's steps 2 to 5 by number, as path.metal.h carries the path
  tracer's: what each pass calls, in order.
- Guidelines: as path.metal.h; P.1 (the algorithm stated where it is
  implemented).

**`metal/integrator/path.metal.h`** (changed)
- Split, not rewritten: the camera's segment to the first rough surface,
  and the path's continuation from a surface, each a function; the path
  pass calls both in turn, as it does now in one, bit-identical; the
  surfaces pass calls the first, the shade pass the second. The reference
  and ReSTIR share every line of light transport but the direct light at
  the first surface (principle 3).
- Guidelines: ES.3 (one light transport, called from three passes); I.7,
  held by a test that the path pass's images are unchanged to the bit.

**`metal/lights/sphere_light.metal.h`, `emitter.metal.h`** (changed;
contract 3)
- New, beside the existing functions: a sample drawn with its identity
  (the point on the sphere it reached), and a sample evaluated from an
  identity at any point: direction, distance, radiance, the cosine at the
  light. sample_light() is unchanged and draws the same numbers.
- Guidelines: I.1 (the identity explicit, never a position kept from
  another frame: principle 6).

**`metal/passes/{surfaces, restir_candidates, restir_temporal,
restir_spatial, restir_shade}/`** (new; Pass)
- One kernel and its launcher each (change-axes.md: a co-located pair).
  What each dispatch reads and writes, its barrier, and its cost.
- Guidelines: GPU.8 (a barrier only between a write and its reader);
  GPU.3 (the split's purpose is fewer registers live across a ray query,
  measured against the counters of 2026-10-10-path-kernel-counters.md, not
  occupancy for its own sake); GDSA.8 (one thread a pixel, by output).

**`metal/frame/frame_images.h`, `accumulation.h`** (changed; contract 6)
- The new images: the surface records, this frame's and last frame's
  (ping-pong); reservoirs: one working, and two finals ping-ponged so last
  frame's survives this frame's spatial pass. Allocated once per size, and
  only for a graph with ReSTIR passes: the path graph allocates what it
  does now.
- Guidelines: MEM.4 (double-buffered across one frame); GPU.9 (transient
  images from the frame's heap); MEM.9.

**`metal/acceleration/scene_acceleration.h`** (added to)
- Last frame's structure kept alive one frame more, for the temporal
  merge's visibility as last frame saw it (RTXDI's `PrevSceneBVH`, used
  when its sample has it). Only for a graph with a temporal pass. A
  structure here is one level of the shapes' placed boxes, rebuilt each
  frame: tens of kilobytes, and the build already paid; what is new is
  holding it, and no frame in flight rebuilding into one another frame
  still reads.
- Guidelines: GPU.7 and GPU.9 (a resource's lifetime tied to the frames
  that read it); MEM.4.

**`metal/scene/shape_transforms.h`, `light_glows.h`** (changed)
- The previous frame's transforms and glows beside this frame's, animated
  at the previous time on the CPU: the temporal merge's targets are
  evaluated as the previous frame lit them. Some 43 µs more a frame, and
  only for a graph with a temporal pass.
- Guidelines: MEM.9; the existing frame-copy rules.

## What it costs, counted

Recounted for the production rework (below), at RTXDI's defaults: M = 8,
H = 20, k = 1, basic mode, permutation and final-visibility reuse on;
logical bytes a pixel per frame (GDSA.6: count passes over memory). The
headers carry the same counts, pass by pass
(`metal/passes/restir_di/restir_di.metal.h` totals them).

| Pass | Reads | Writes | Rays | Target evaluations |
|---|---|---|---|---|
| restir_initial | | 32 record + 36 path state + 12 previous position + 24 reservoir = 104 | camera's, as now; 1, visibility reuse | M = 8 |
| restir_temporal | 32 + 12 + 24 + 32 + 24 = 124 (+ 32 a further pixel tried, up to 256 more at a disocclusion) | 24 | none (raytraced: at most 1) | at most 2 |
| restir_spatial | (1 + k) x 56 = 112 with its second walk cached, (1 + 2k) x 56 = 168 without | 24 | none (raytraced: at most k) | at most 2k = 2 |
| restir_shade | 32 + 36 + 24 = 92, and the accumulated pixel's 16 | the accumulated pixel's 16, and at most 8 back to the reservoir where it traced | 0 or 1 (raytraced: 0), then the continuation's | 1 |

In all, 512 to 576 bytes a pixel of intermediate traffic, against 1368 to
1928 in 4463eee's design (its 80-byte record, 32-byte reservoir, 64-byte
path state, five neighbors and their second walk). At a traced 1728 x 1117
that is 0.99 to 1.11 GB a frame: at the M3 Max's peak 400 GB/s, an
ideal-bandwidth floor of 2.5 to 2.8 ms were none of it cached, the least
the traffic could take, not the most (corrected 2026-10-11, by the review
of f86e7ae). Light data: at most M + 2k + 3 = 13 target
evaluations, 988 logical bytes a pixel, assumed cached (76 KB for a
thousand fireflies). Rays at the first rough surface: 1 to 2 in basic mode,
k + 2 = 3 in raytraced, against the naive estimator's 1. ReuseImages: 184
bytes a pixel, 355 MB at 1728 x 1117. Counted, not measured.

The goal the headers state, in two parts, both for the implementation to
measure: ReSTIR's added cost per frame in line with production
implementations, a few milliseconds at the traced resolution and not a
multiple of the naive frame (a ReSTIR frame does cost more than a naive one
at the same samples per pixel: its candidates, its records' and reservoirs'
traffic, its visibility ray and its merges); and much lower error x time
than the naive estimator, MSE-seconds by `make convergence`, the comparison
the demo makes. The naive baseline it is measured against: 29.90 ms at
3456 x 2234 and 1.3 to 3 MSE-seconds (2026-10-10-reference.md).

## Choices, and what was set aside

| Choice | Kept | Set aside, and why |
|---|---|---|
| Where | The first rough surface | Every bounce (ReSTIR PT): a later milestone |
| Bias correction | `basic`, RTXDI's `Basic`, the default: the inputs' weights from their unshadowed targets, no rays in the merges; and `raytraced` as a setting, for checking ReSTIR against the reference | The paper's unbiased merges as the default (every domain's visibility ray and the own-domain ray, k + 2 rays a pixel): exact, and with the other textbook choices below it made a ReSTIR slower than the naive path tracer at full resolution. Real-world software accepts Basic's darkening at shadow edges; the user directed it on 2026-10-10. Kept as `raytraced`. Earlier set aside: no visibility reuse (the paper's gain in an occluded scene lost); pairwise MIS (with visibility in the target, 2k rays against 1/Z's k) |
| Normalization's weights | RTXDI's MIS-like weights in both modes: W = w_sum pi / (p̂_q(y) piSum), pi the source's domain target at y, piSum the sum of each input's count times its domain target, unshadowed in `basic`, with visibility in `raytraced`; so the modes differ in rays alone and `raytraced` checks the code `basic` ships. The source's pi is its reservoir's stored target, no evaluation | 1/Z, the paper's Algorithm 6, an input counted by its whole count where it could have produced y (1c19b2a, and 4463eee before it): unbiased too, but it shares each sample equally among the domains that could have made it, however unlikely one was to, which costs noise where the inputs' targets differ, at the same evaluations and rays as the weights RTXDI ships. Set aside 2026-10-10 by the user's direction, production's defaults now, not a later measured change |
| Candidates | M = 8 lights, uniform (RTXDI's `Uniform` local light sampling) | RTXDI's 1 BSDF candidate: a 1.5 mm firefly is almost never hit by a BSDF sample, and the candidate would cost a ray; power-weighted selection: later, measured |
| Light identity | Index with a valid bit and a point in 2 x 16 bits, equal-area octahedral for a sphere: 8 bytes, RTXDI's `lightData` and `uvData` | The point as drawn, three floats (16 bytes; 4463eee): exact, where the quantization is a small bias, of the order of a cell, 10^-4 of the light's radius; production accepts it, and the user directed it on 2026-10-10. It is why `raytraced` is a check within a tolerance, which the implementation measures. Cigolle et al.'s plain octahedral map: cells of unequal area on the sphere; equal-area makes the quantization the same everywhere on a light, for a square root more to decode. A position: wrong next frame (principle 6) |
| Reservoir | 24 bytes, RTXDI's six words: sample (8), M (24 bits), where and when its visibility was tested (offset 2 x 8 bits, age 8), its target at its own surface, W | 32 bytes with 8 of padding (4463eee), so no reservoir straddles a cache line in the spatial gathers: a third more traffic on every read, to spare the 1 in 8 gathered reservoirs that straddle a 128-byte line, one gather a pixel at k = 1; set aside 2026-10-10 by the same direction. RTXDI's 18 bits of RGB visibility: shadows here are binary and a hidden sample weighs 0, so W > 0 is that bit |
| Surface record | 32 bytes, G-buffer sized: position at full precision (through glass and mirrors the surface is not on the camera ray, so a depth cannot place it), wo packed (4), contract 2's PackedBsdf (16: normal packed, halves) | 80 bytes, contract 2's Bsdf whole, position and wo at full precision (4463eee): exact, and most of the merges' traffic. Production's G-buffers pack; the target and the shade read the same unpacked surface, so the packing is a small, stated difference from the reference's surface (normal and wo within 10^-4 radians, parameters within 5 x 10^-4 relative), not a mismatch between target and integrand. Primitive and position only, the BSDF re-resolved per neighbor: texture lookups in every merge |
| Path state | 36 bytes, throughput, L and eta scale at full precision, counters and media in 16-bit fields | 64 bytes (4463eee), with the geometric normal, which equals the shading normal for every surface so far |
| Spatial neighbors | k = 1 within 32 pixels, geometry and material tests (RTXDI's `numSamples`, `samplingRadius`, `enableMaterialSimilarityTest`) | k = 5 within 30 pixels (the paper's, 4463eee): five gathered records and reservoirs a pixel, and in raytraced mode five rays. k stays a setting |
| Temporal lookup | The reprojected pixel, rounded to nearest, permuted within its 4 x 4 block each frame (RTXDI's permutation sampling), then up to 8 pixels around it until one is similar (RTXDI tries up to 9) | One pixel, rounded down (the paper's, 4463eee): a disocclusion found no history where a pixel beside it had one. RTXDI draws the further tries at random; a fixed ring draws no numbers |
| Final visibility | The shade traces its survivor's ray, unless its reservoir's test is this pixel's own this frame, or, in basic mode, within 4 frames and 16 pixels (RTXDI's `reuseFinalVisibility`, `finalVisibilityMaxAge`, `finalVisibilityMaxDistance`); a traced result written back, a hidden sample's W set to 0 | No shade ray, by raytraced mode's invariant (4463eee): bought with k + 1 rays in the merges. Always tracing: a ray a pixel that a recent test answers; the reuse's bias, light leaking for up to 4 frames or 16 pixels at shadow edges, is production's |
| Boiling filter | Not in the first version; named as a later, measured addition | RTXDI's default (strength 0.2): at the end of the temporal merge, a threadgroup's reservoirs whose weights stand far above the group's are emptied. A deliberate bias, against artifacts ("boiling") this scene has not yet been seen to show; and the first threadgroup-wide reduction and barrier in these kernels (GPU.5), whose result depends on the dispatch's group shape. Added when the first version's images show boiling, with its strength a setting, and measured |
| Other RTXDI defaults not carried | | `discountNaiveSamples` (neighbors of small M skipped; its threshold is not in the headers read); the disocclusion boost (8 neighbors where history is short), off by default (`targetHistoryLength = 0`); its table of neighbor offsets, clamped to the image (here a disc draw, refused outside the image); checkerboard rendering |
| Kernels | Four passes | One megakernel: the spatial merge needs every pixel's reservoir first, a barrier between |

## Changes to the architecture documents

- logical-overview.md describes this algorithm and its rays; it gains
  visibility reuse's one ray beside them, and, with the production rework,
  both modes' ray counts and principle 3 restated for a default that is
  production's, with an unbiased mode beside it.
- file-mapping.md gains contracts 4, 5 and 14 in its table, and contract
  2's packed form.

## Decided (2026-10-10)

- At the first rough surface (RTXDI's primary surface replacement through
  glass and mirrors), naive next event estimation at every later bounce
  until ReSTIR GI.
- **The production rework, later the same day.** Serenity is an IC demo
  that explores the parallel between attention and path tracing and has to
  perform like real-world software; it is not a paper. 4463eee kept
  choosing the textbook-exact option (the ray-traced unbiased merges, five
  neighbors, exact float light points, a 32-byte reservoir, an 80-byte
  surface record), and together those gave a ReSTIR of some 1,400 to 1,900
  bytes a pixel and k + 2 rays, slower than the naive path tracer at full
  resolution. The user decided: the default configuration is RTXDI's
  shipped production defaults (its ReSTIRDI.cpp: initial visibility on,
  Uniform local light sampling, 8 local light samples; temporal Basic,
  history 20, permutation sampling on, boiling filter on at 0.2; spatial
  1 sample, Basic, radius 32, material similarity on, depth 0.1, normal
  0.5; final visibility on and reused, at most 4 frames and 16 pixels),
  and the unbiased ray-traced merges stay only as a setting, for checking
  against the reference. "Choices, and what was set aside" above records
  each textbook-exact choice replaced, and what of RTXDI's defaults is not
  carried, with why (the boiling filter, a later and measured addition).
  Supersedes this list's first entries: the paper's unbiased merges as the
  default; five passes and a 64-byte record; k = 5 and a 30-pixel radius.
- Defaults, data, varied by measurement: M = 8, H = 20, permutation on,
  k = 1, a 32-pixel radius, thresholds 0.1 and 0.5, bias correction basic,
  final-visibility reuse on at 4 frames and 16 pixels
  (core/passes/restir_di.h).
- The goal, two parts, each measured by the implementation: ReSTIR's added
  cost per frame in line with production, a few milliseconds at the traced
  resolution, not a multiple of the naive frame; and much lower error x
  time than the naive estimator (`make convergence`). A ReSTIR frame is
  expected to cost more than a naive frame at the same samples per pixel;
  its win is far less error at equal time.

## Where the headers depart from this layout (2026-10-10)

Written while designing the headers, and revised after Codex's review of
70dbe59; each header states the choice, and this note keeps what was set
aside. The guideline pass over the headers, and the review's findings
with what each changed, are in
`2026-10-10-restir-di/headers-guidelines-pass.md`. The rows marked
(rework) record the production rework (Decided, above), each against the
4463eee headers it replaces; the earlier rows they name stand as the
history of the choice, superseded.

| Layout | Headers | Why |
|---|---|---|
| Five passes: surfaces, candidates, temporal, spatial, shade | Four: `restir_initial` (steps 1 to 3: the camera segment, M candidates, visibility reuse), temporal, spatial, shade | Surfaces and candidates are adjacent, always together, and share no pixel: split, they wrote the record and read it back (80 bytes a pixel) across a dispatch and a barrier. The path state is written before the candidates, so what is alive across visibility reuse's query is the record and the reservoir, as in a separate kernel (review of 70dbe59) |
| `metal/integrator/restir_di.metal.h`, the steps in the Integrator | `metal/passes/restir_di/restir_di.metal.h`, a Pass-axis composition the four passes include | In the Integrator it included Sample reuse, which change-axes.md rules out. The merges now ask a Frame the composition passes them for the target, visibility and depths, and include no other family's shaders but the Sampler's numbers; the Integrator knows no reservoir (review) |
| The shade traces one shadow ray; k + 3 rays a pixel | No shade ray; k + 2 | The merges keep an invariant: every reservoir written holds a sample visible from its surface, or weighs 0. The temporal merge's ray, when the survivor came from last frame, is traced from this pixel's surface, and the spatial merge's, when it came from a neighbor, likewise: each tests the pixel's own domain for Z and is the visibility the shade needs. Without it (the layout's plan, and RTXDI's), Z counts the pixel's own candidates for a sample it cannot see; harmless to its own image, which the final ray zeroes, but the merged reservoir then holds samples outside the domain a neighbor's spatial test grants it, a bias in that neighbor's image (`metal/reuse/unbiased.metal.h`) |
| A light point as the light's index and an octahedral point in 2 x 16 bits: 8 bytes | The index and the point as drawn, three floats in the light's own space: 16 bytes | A point snapped to a grid makes the sample the cell's representative, whose probability is the density integrated over the cell; weighed by the density at the point, the estimator sums a quadrature that no number of frames turns into the reference's integral. The first headers stated the error as small; it is a bias, and both reviews of 70dbe59 said so. Kept exactly, the estimator integrates the reference's measure |
| A reservoir of 16 bytes | 32: the 16-byte light point, W, M, and 8 bytes of padding | Its fields fill 24. At 32 a reservoir never straddles a cache line and loads as two aligned 16-byte words, which the spatial merge's gathers, one reservoir a lane from unrelated places, are where it pays; the padding is counted in every pass's traffic |
| The record: position, wo octahedral (4 bytes), Bsdf (48): 64 | Contract 2's Bsdf whole, position and wo at full precision, two words of padding: 80 | A BSDF evaluated at a quantized wo is the image of another surface. The first headers kept 64 by copying Bsdf's fields without its padding; that defined Material's layout twice, and a field Material added in its padding would have been dropped silently (review). No layout of one Bsdf and the two full-precision triples fits 64 |
| The temporal merge finds where the surface was from its position through the previous camera | A third array, the surface's previous position (12 bytes, temporal graphs only), written by the initial pass from the hit's object-space point and last frame's transform | The temporal merge cannot place a moving shape's point from the record alone, and the spatial merge's k neighbor reads should not carry it |
| Reuse constants: the previous camera and time | The previous camera; no time | No shader reads the previous time: last frame's transforms and glows, animated at it on the CPU, are what carry it |
| Target evaluations: temporal 2, spatial 2k | 3 and 1 + 2k | The pixel's own reservoir is streamed with its sample's target, which the reservoir does not store |
| About (5 + k) x 80 bytes a pixel, 800 at k = 5 | Intermediate-buffer traffic 1368 to 1928 bytes a pixel at k = 5; light data counted apart, 76 logical bytes per target evaluation, 1748 a pixel at the written settings, assumed cached; traversal counted as rays | Counted from the passes: the path state, the previous position, the accumulated pixel, the 80-byte record and 32-byte reservoir, and the spatial merge's second walk over its neighbors for Z (from cache or not). The first headers' total of 1080 to 1480 was intermediate traffic alone, without saying so (review) |
| Last frame's structure "kept alive one frame more" | A ring of three structures, indexed by frame index, for a temporal graph whose shapes move, each with the last submission that traced it | With one per slot, frame i + 1 rebuilds what frame i reads as last frame's. The first headers promised three and declared storage for two (review) |
| The twelve files | Also: `metal/reuse/similar.metal.h` (the similarity test both merges ask), `metal/camera/thin_lens.metal.h` (projection and depth), `metal/sampler/sampler.metal.h` (a purpose's stream), `metal/light_selection/uniform_light.metal.h` (the chosen index), `core/frame/frame_inputs.h` (reused_since), `core/frame/schedule.cpp` (its list asserted against the final kind), `metal/frame/frame_resources.h` and `renderer.h` | Each is the axis that owns the piece; every change is a declaration or a comment beside what exists, or, in `schedule.cpp`, an assertion that now checks the whole enum |
| (rework) Merges: the paper's unbiased normalization with every domain's ray and the own-domain ray; no shade ray; k + 2 rays | `bias_correction`: `basic` by default, no ray in a merge, the shade's ray or a reused test, 1 to 2 rays; `raytraced`, the 4463eee design kept whole, k + 2 | RTXDI's production default (user's direction, 2026-10-10). `metal/reuse/unbiased.metal.h` states both modes, their rays and basic's bias: darker on the lit side of shadow edges, where Z counts an input that cannot see the survivor. Both modes use RTXDI's MIS-like weights, pi over piSum (1c19b2a divided by Z; this table's last row says why that changed), so they differ in rays alone |
| (rework) The light point as drawn, three floats: 16 bytes | The index with a valid bit, and the point equal-area octahedral in 2 x 16 bits: 8 bytes (RTXDI's `lightData`, `uvData`); its map declared again in `metal/math/warp.metal.h` (`encode_unit`, `decode_unit`), shared by the record's wo and normal | The quantization's bias, of the order of a cell (about 10^-4 of the light's radius), is stated in `contracts/light_sample.h` as production accepts it; it makes `raytraced` a check within a tolerance, which the implementation measures. Equal-area over Cigolle's octahedral: every cell the same area, in the measure the estimator integrates |
| (rework) Reservoir: 32 bytes, 8 of them padding | 24 bytes: the sample (8), M in 24 bits, `tested` (offset 2 x 8 bits and age 8 bits), the target at its own surface, W | RTXDI's six words. The stored target is the earlier "set aside" use of the padding, now taken: each merge streams its own input by it, and takes a source's as that source's pi, which saves the evaluations the 4463eee row "3 and 1 + 2k" added. RTXDI's visibility bits are not kept: binary shadows and W = 0 for a hidden sample make W > 0 that bit |
| (rework) Record: contract 2's Bsdf whole, position and wo at full precision: 80 | 32: position (12), wo encoded (4), contract 2's `PackedBsdf` (16: the normal encoded, color and the kind's scalar and escape as halves, the kind), owned by Material in `contracts/bsdf.h` with `pack_bsdf` and `unpack_bsdf` (`metal/materials/bsdf.metal.h`) | One definition kept (the Codex finding on 70dbe59): the packed form is Material's, beside the Bsdf. Not exact, and stated: every pass, the shade included, reads the same unpacked surface, so the target is the integrand without visibility exactly, and the surface differs from the reference's by the packing's error (`contracts/surface_record.h`) |
| (rework) Path state: 64 bytes | 36: throughput, eta scale and L in floats; the numbers' counter, the surface's count and contract 1's flags in one word; the two media in 16 bits each | Read once a pixel by the shade, so exact where the image depends on it and packed where the values are small; the geometric normal dropped, equal to the shading normal for every surface so far; a scene of 0xffff media or more refused by the renderer |
| (rework) Spatial: k = 5 within 30 pixels | k = 1 within 32, and the material test (`metal/reuse/similar.metal.h`, `similar_material`, tolerances named there) | RTXDI's defaults; k a setting. The material test is RTXDI's spatial one; its comparison is RTXDI's application's, so Serenity's tolerances are its own |
| (rework) Temporal: one pixel, rounded down | Rounded to nearest, permuted within its aligned 4 x 4 block by the frame's two numbers (pixel (0, 0)'s stream, so no CPU value), then the 8 pixels around until one passes | RTXDI's permutation sampling and its up-to-9 tries; its further tries are random, a fixed ring here |
| (rework) The shade writes nothing | In a graph with a temporal merge, where it traced, its own pixel's final reservoir: the test, and W = 0 where hidden | So the next frame's final-visibility reuse has a test to take, and does not carry a sample q is known not to see |
| (rework) Last frame's structure kept for every temporal graph | Only for a raytraced one | Basic mode's merges trace no ray, and a temporal graph in it keeps nothing more than last frame's transforms and glows |
| (rework) Traffic 1368 to 1928 bytes a pixel at k = 5; 23 target evaluations; k + 2 rays | 512 to 576 bytes; at most 13 evaluations; 1 to 2 rays in basic, 3 in raytraced, at the defaults | Counted pass by pass in the headers ("What it costs, counted", above) |
| (rework) Reuse constants: 112 bytes | 128: the settings grew to 48 bytes (mode, permutation, visibility reuse and its two limits) | The settings a shader reads, still one definition (`core/passes/restir_di.h`) |
| (rework, after 1c19b2a) Normalization: 1/Z, the inputs that could have produced y counted by their whole counts | RTXDI's pi over piSum in both modes (`metal/reuse/unbiased.metal.h`): pi the source's domain target, read from its reservoir's stored target; piSum each input's count (the temporal one capped at H times the pixel's) times its domain target, unshadowed in basic, times visibility in raytraced. The same evaluations and rays as 1/Z | Read from RTXDI's temporal and spatial resampling: both modes, and the canonical input weighted by its stored target and its own unshadowed target in piSum. The same target evaluations, rays and memory as 1/Z, not the same instructions (a count-times-target sum and a multiply where 1/Z adds counts); less noise is the expectation, not a measurement, until `make convergence` compares them. Production's default, adopted now on the user's direction rather than deferred. Two departures, stated in the header: the source's visibility is not traced (the invariant gives it), and raytraced mode traces the canonical input's (RTXDI's does not) |

Set aside in designing them: aliasing the frame-transient arrays (the
working reservoirs, the path states) with the tone map's pyramid in one
heap (GPU.9), until memory at the traced resolution is a constraint; and a
64-byte record, which no single definition of the Bsdf and two
full-precision directions fits (superseded by the rework's 32-byte packed
record). The reservoir's stored target, set aside here for 4463eee, is
taken in the rework.

## Phase A, implemented (2026-10-10)

ReSTIR DI without reuse, end to end: the path tracer's split, contracts
4, 5 and 14, light points, the target, the reservoir's candidates and
finish, `restir_initial` and `restir_shade`, `graphs/restir_ris.toml`. The
merges, the reuse history, last frame's copies and structure, raytraced
mode's rays and the boiling filter are Phase B's; the Metal backend
refuses `restir_temporal` and `restir_spatial` by name until then. Where
the headers changed in implementing them, and why:

| Header | Change | Why |
|---|---|---|
| `metal/reuse/reservoir.metal.h` | `Candidate` carries the weight p̂ / p, not the density; `stream_candidate` returns whether it kept the candidate | The target and the density share cos θ_y a / d², which the target header says the implementation cancels; the reservoir is given the quotient (`candidate_weight`). The return lets the candidates keep the survivor's light answer for visibility reuse's ray without evaluating it again (F.21) |
| `metal/integrator/target.metal.h` | `candidate_weight(surface, light, density)` added, with `scattered()` and `area_factor()` | The cancelled form, defined once (ES.3); one density argument (I.24) |
| `core/contracts/light_sample.h`, `surface_record.h`, `bsdf.h` | `unit_round_trip_error` = 10^-4 radians, the map's measured worst 6.3 x 10^-5 stated where "about 5 x 10^-5" was | 5 x 10^-5 is a cell's width; a round trip's worst, where the map shears, measured over 262,336 directions, is 6.3 x 10^-5 |
| `metal/math/warp.metal.h` | r = sqrt((x² + y²) / (1 + |z|)) in place of sqrt(1 - |z|) | The same number for a unit vector, without cancelling near the poles, where sqrt(1 - |z|) lost several cells |
| `core/contracts/bsdf.h`, `surface_record.h` | the packed fields' shifts and masks named | ES.45 |
| `metal/frame/frame_images.h` | `ReuseImages` makes a graph without merges' arrays (one record array, the path states, one final array) and refuses `Temporal::merged` and `Working::needed` by `MetalError`; no history member yet | Only the arrays Phase A's passes use; `ReuseHistory` arrives with the temporal merge |
| `metal/frame/renderer.h` | the argument table holds 16 buffers for a ReSTIR graph, 8 as before for any other; the queue barrier also precedes the first ReSTIR pass | Its passes bind up to 12 (16 with the temporal merge); additive |
| `core/frame/schedule.h` | `Schedule::restir_di` defaults to none | So a schedule written {passes, tone_map} keeps its meaning and the existing tests compile unchanged |
| `metal/passes/restir_di/restir_di.metal.h` | `takes_test()` (step 6's rule), `direct_from()`, `pack_medium`/`unpack_medium` added; `FrameView`, `frame_permutation` marked Phase B; the shade's write-back deferred to the temporal merge | No graph that runs reads the write-back yet; the shade binds the final reservoirs read-only |
| `metal/passes/restir_initial/restir_initial.h`, `restir_shade.h`, `core/passes/restir_di.h` | what Phase A refuses or defers, stated | No stubs that look real |
| `metal/integrator/path.metal.h` | the loop as `arrive()` (steps 1 to 4) and `scatter()` (6 to 8) over a `Walk` | So the segment and the continuation share every line; the path pass's image unchanged to the bit (phase-a-guidelines-pass.md) |
