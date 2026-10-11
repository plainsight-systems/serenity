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
3. **Temporal merge.** The reservoir this surface held last frame, found
   where the surface was then, merged in, its history capped.
4. **Spatial merge.** The reservoirs of k neighbors within a radius,
   merged in.
5. **Shade.** One shadow ray toward the survivor; the direct light is
   f |cos| L_e G V x W, W the reservoir's contribution weight. The path
   then continues from the surface as the naive path tracer does, for the
   indirect light, and the pixel's radiance is folded into the accumulated
   image.

Merges 3 and 4 weigh each reservoir by generalized RIS with pairwise MIS
(Bitterli 2022, thesis ch. 9.1; Wyman et al. 2023, "A Gentle Introduction
to ReSTIR"), the bias correction RTXDI offers as `Pairwise`: O(k) target
evaluations, not O(k²). The target is unshadowed throughout, so no merge
traces a ray and the estimator stays unbiased; visibility enters once, at
step 5. Reusing visibility (RTXDI's ray-traced correction) is a later,
measured option.

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
- The reservoir in registers and its three operations: stream a candidate
  (w = p̂ / p, keep it with probability w / w_sum), merge another reservoir
  with a given MIS weight, and finish (W = w_sum / p̂(y)). The one merge
  rule every reuse uses, direct and later indirect (change-axes.md).
- The softmax parallel, and the test that holds it (above).
- Guidelines: CDSA.23 (update and merge, a fixed order); GDSA.21 (the same
  draw as the random-key race); GDSA.3 (each choice's number from pixel,
  frame and purpose, metal/sampler).

**`metal/reuse/pairwise_mis.metal.h`** (new; Sample reuse)
- Defensive pairwise MIS with confidence weights (Bitterli 2022, eq. 9.10,
  Alg. 8): each neighbor weighed against the pixel's own (canonical)
  reservoir, the canonical's weight renormalized after the last neighbor.
  Four target evaluations a neighbor.
- Guidelines: GDSA.6 (O(k) evaluations, not O(k²)); I.7 (the weights sum
  to 1 over the domains for any sample: the unbiasedness condition, a test).

**`metal/reuse/temporal.metal.h`, `metal/reuse/spatial.metal.h`** (new;
Sample reuse)
- Temporal: where this surface was last frame (its position through the
  previous camera), whether that pixel's surface is the same one (depth
  within 10%, normals within cos 0.5), the history cap H x M. A surface
  seen through glass reprojects wrongly; the similarity test and the MIS
  weights make that noise, never bias (logical-overview.md).
- Spatial: k neighbors drawn within the radius, each tested as above.
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
| candidates | 64 P | 16 P | none; M target evaluations |
| temporal | 2 x 64 P + 2 x 16 P | 16 P | none; 4 target evaluations |
| spatial | k x (64 + 16) P | 16 P | none; 4k target evaluations |
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
| Bias correction | Pairwise MIS, unshadowed target, no rays in merges | 1/Z with visibility rays (Bitterli 2020's unbiased form): k + 1 rays more a pixel, to be measured as an option; no correction: biased |
| Candidates | M lights, uniform | A BSDF candidate (RTXDI's 1): a 1.5 mm firefly is almost never hit by a BSDF sample; power-weighted selection: later, measured |
| Light identity | Index + octahedral point, 8 bytes | A position: wrong next frame (principle 6) |
| Surface record | Resolved BSDF, 64 bytes | Primitive and position only, the BSDF re-resolved per neighbor: texture lookups (the wood's noise) in every merge |
| Kernels | Five passes | One megakernel: the spatial merge needs every pixel's reservoir first, a barrier between |

## Changes to the architecture documents

- logical-overview.md says an unbiased ReSTIR pixel traces about one ray
  for the temporal merge and k for the spatial. With pairwise MIS over an
  unshadowed target, the merges trace none; the overview is corrected when
  the headers are.
- file-mapping.md gains contracts 4, 5 and 14 in its table.

## Open, for the user

1. Pairwise MIS without visibility rays as the first version, with
   visibility reuse as a measured option later?
2. Five passes and a 64-byte surface record first, packing later if
   measured?
3. Defaults: M = 8, H = 20, k = 5, radius 30 px?
