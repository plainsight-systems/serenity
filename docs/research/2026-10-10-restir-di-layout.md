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

Merges 4 and 5 are the paper's unbiased combination (Bitterli et al. 2020,
Algorithm 6). Each input reservoir r_i, from surface q_i, is streamed with
weight p̂_q(y_i) W_i M_i, p̂_q this pixel's target; the survivor y's
weight is then W = w_sum / (Z p̂_q(y)), where Z sums the counts M_i of the
inputs whose surface could have produced y: those where p̂_{q_i}(y) > 0
and y is visible from q_i. Visibility reuse (step 3) put visibility into
what every reservoir holds, so it must be in that test too: one shadow ray
from each input's surface toward y. The temporal input's surface is last
frame's, and is tested as last frame saw it: its lights where they were,
traced against last frame's acceleration structure. Without the rays, the
merges are biased (darker, in shadow, RTXDI's `Basic`); with them,
unbiased (RTXDI's `Raytraced`), the reference holds ReSTIR to the same
image, and MSE x time charges it for every ray.

Rays a pixel: 1 for visibility reuse, 1 for the temporal merge's test, k
for the spatial merge's, 1 to shade: k + 3, 8 at k = 5, against the naive
estimator's one at that surface.

How production does it: NVIDIA's RTXDI library splits ReSTIR DI the same
way (initial sampling, temporal, spatial, pairwise MIS, reservoir storage),
stores a light by index and a point on it (`lightData`, `uvData`), and
defaults to 8 light candidates, a history of 20, a spatial radius of 32
pixels, a depth threshold of 10% and a normal threshold of cos 0.5. Its
library is under NVIDIA's proprietary license: read for its design, none
of it copied.

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

At P pixels, k neighbors, per frame (GDSA.6: count passes over memory):

| Pass | Reads | Writes | Rays |
|---|---|---|---|
| surfaces | | 64 P + path state | the camera's, as now |
| candidates | 64 P | 16 P | 1, visibility reuse; M target evaluations |
| temporal | 2 x 64 P + 2 x 16 P | 16 P | 1, Z's test; 2 target evaluations |
| spatial | k x (64 + 16) P | 16 P | k, Z's tests; 2k target evaluations |
| shade | 64 P + 16 P + path state | radiance | 1 shadow ray + the continuation |

About (5 + k) x 80 bytes a pixel, 800 at k = 5: 6 GB a frame at 3456 x
2234, some 15 ms at 400 GB/s, and 1.5 GB (4 ms) at half that each way. The
surface record is most of it. Kept at 64 bytes in the first version
because it re-evaluates the target exactly, with no texture lookups in the
merges; packing the BSDF (to some 16 bytes) is the first measured follow-up
if the merges are bound by it. The baseline it is measured against: 29.90
ms and 1.3 to 3 MSE-seconds (2026-10-10-reference.md), by `make
convergence` on a ReSTIR graph.

## Choices, and what was set aside

| Choice | Kept | Set aside, and why |
|---|---|---|
| Where | The first rough surface | Every bounce (ReSTIR PT): a later milestone |
| Bias correction | The paper's unbiased 1/Z with visibility rays, and visibility reuse | No rays in the merges and no visibility reuse: unbiased too, but neighbors share lights hidden behind marbles, and the paper's gain in an occluded scene is lost (proposed in this note's first draft, set aside the same day); `Basic` (no rays, visibility reuse kept): biased, darker in shadow; pairwise MIS (Bitterli 2022, O(k)): with visibility in the target it needs 2k rays, against 1/Z's k |
| Candidates | M lights, uniform | A BSDF candidate (RTXDI's 1): a 1.5 mm firefly is almost never hit by a BSDF sample; power-weighted selection: later, measured |
| Light identity | Index + octahedral point, 8 bytes | A position: wrong next frame (principle 6) |
| Surface record | Resolved BSDF, 64 bytes | Primitive and position only, the BSDF re-resolved per neighbor: texture lookups (the wood's noise) in every merge |
| Kernels | Five passes | One megakernel: the spatial merge needs every pixel's reservoir first, a barrier between |

## Changes to the architecture documents

- logical-overview.md already describes this algorithm and its rays (one
  for the temporal merge, k for the spatial, one to shade); it gains
  visibility reuse's one ray beside them.
- file-mapping.md gains contracts 4, 5 and 14 in its table.

## Decided (2026-10-10)

- The paper's ReSTIR DI, as written: visibility reuse, and the unbiased
  merges' rays. A first draft of this note dropped both to keep the merges
  free of rays; that was a convenience, not the algorithm, and was set
  aside.
- At the first rough surface (RTXDI's primary surface replacement through
  glass and mirrors), naive next event estimation at every later bounce
  until ReSTIR GI.
- Five passes and a 64-byte surface record first; packing it is the first
  measured follow-up.
- Defaults M = 8, H = 20, k = 5, a 30-pixel radius, thresholds 0.1 and 0.5:
  data, varied by measurement.

## Where the headers depart from this layout (2026-10-10)

Written while designing the headers, and revised after Codex's review of
70dbe59; each header states the choice, and this note keeps what was set
aside. The guideline pass over the headers, and the review's findings
with what each changed, are in
`2026-10-10-restir-di/headers-guidelines-pass.md`.

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

Set aside in designing them: aliasing the frame-transient arrays (the
working reservoirs, the path states) with the tone map's pyramid in one
heap (GPU.9), until memory at the traced resolution is a constraint; using
the reservoir's 8 bytes of padding to keep its sample's target at its own
surface, which would save the merges one target evaluation each, until the
counters say target evaluations matter; and a 64-byte record, which no
single definition of the Bsdf and two full-precision directions fits.
