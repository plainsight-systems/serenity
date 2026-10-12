# ReSTIR PT: the design

2026-10-11. ReSTIR DI is built (Phases A to C, docs/research/
2026-10-11-restir-spatial.md) and loses to the path tracer on the marbles at
equal time: it reaches only the light that arrives straight from a firefly
at the first rough surface, and most of this scene's light arrives through
glass and off mirrors. This note designs ReSTIR PT: reuse of whole paths,
the variant whose reuse reaches through glass and mirrors. It starts from
the scene, then the production defaults, then the headers.

Sources, read 2026-10-11:

- Lin, Kettunen, Wyman, "ReSTIR PT Enhanced: Algorithmic Advances for
  Faster and More Robust ReSTIR Path Tracing", PACMCGIT 9(1), May 2026, and
  its supplemental (the PDFs, kept out of the repo). Cited below as
  "Enhanced §n".
- Lin et al. 2022, "Generalized Resampled Importance Sampling: Foundations
  of ReSTIR", and its Falcor source, github.com/DQLin/ReSTIR_PT
  (ReSTIRPTPass.h, Params.slang, PathReservoir.slang). Cited as "the 2022
  code".

No shipped product runs ReSTIR PT; RTXDI ships DI and GI. Enhanced is
NVIDIA's step toward production ("closer to production-ready", its
abstract), built on the 2022 code, so its defaults are the production
defaults here.

## The scene first

The marbles (scenes/marbles.toml): 36 marbles 1.2 to 2.5 cm across on a
plank table, seen from 39 cm, black sky, lit only by 616 fireflies 1.5 mm in
radius. Its light, by the path it takes to the camera, and what each ReSTIR
reaches:

| What the viewer sees | Path (camera ... light) | ReSTIR DI | ReSTIR PT |
|---|---|---|---|
| Porcelain and table lit by a firefly | rough → light | yes | yes (Enhanced §6.1 folds direct light in) |
| The porcelain's coat: each firefly a sharp point | glossy coat → light | poorly (Phase A's spikes) | yes: lobe-indexed paths, the coat lobe postponed, the base reconnected (supplemental §1) |
| The night refracted upside down in clear and tinted glass | glass → glass → rough → light | no | yes: random replay through the glass, reconnection at the rough surface behind |
| Brass, steel and copper mirrors | mirror → rough → light | no | yes: as glass |
| Bounce light on the table and porcelain | rough → rough → light | no | yes: reconnection at the second rough vertex |
| Cat's-eye cores, lit only through the shell | glass → core → glass → light | no | partly: no reconnection is allowed beside glass, so only full replay, which a neighbor's shifted core point mostly misses at a 1.5 mm firefly; temporal reuse keeps a found path while the camera and firefly are still |
| Caustics on the table | rough → glass → glass → light | no | partly: as the cores. The camera is fixed in this scene, so temporal reuse is a near-identity shift and a found caustic path persists up to the confidence cap |

So ReSTIR PT reaches every kind of light in the frame, fully for all but
the last two. Those two are what manifold next event estimation is for
(scenes/marbles.toml says so; docs/process/QUEUE.md), and stay there.

## What is built: Enhanced at its defaults

| Piece | Default | Source |
|---|---|---|
| Initial samples | 1 path tree per pixel, NEE at every vertex x1 ... xn, one path chosen by RIS over the tree | Enhanced §2.3, §7 |
| Direct light | in the same reservoir (paths of length 2 compete with longer ones) | Enhanced §6.1 |
| NEE candidates | RIS at each vertex: 32 at x1, 32 / B² (at least 1) at bounce B, uniform over the 616 lights, visibility tested for the chosen one only | Enhanced §6.1, §7 |
| Path length | at most 9 surfaces | 2022 code, maxSurfaceBounces |
| Russian roulette | at initial sampling only, never in replay; survival folded into the source pdf | Enhanced §6.2.4, supplemental §6 |
| Lobes | one sampled lobe per vertex, NEE over all lobes | supplemental §1 |
| Shift | hybrid: random replay until the first pair (x_{k-1}, x_k) passing the criteria, reconnect there | Enhanced §2.3 |
| Reconnection criteria | dual footprint, eq. 5, c = 0.02; roughness at x_{k-1} only, α ≥ 0.2; the footprint test at x_k skipped for diffuse or emissive x_k | Enhanced §4 |
| Forced NEE reconnection | a replayed path ending at an NEE light reconnects to it if nothing earlier did | Enhanced §6.2.3 |
| Replay compaction | pixel-neighbor pairs that need replay compacted before replay | Enhanced §6.2.2 |
| Temporal reuse | confidence cap 20, Talbot MIS | 2022 code; Enhanced §7 |
| Decorrelation | duplication map, 17 × 17, D = count / 288; c_cap = lerp(20, 1, D^0.1) | Enhanced §5 |
| Spatial reuse | 3 neighbors, paired (reuse textures 254, 230, 210 wide, σ = 16, the match for R = 30), pairwise MIS; each texture flipped, mirrored, transposed and offset per frame | Enhanced §3, §7 |
| Color noise | vector-valued resampling weights summed for shading | Enhanced §6.3 |
| Reservoir | 64 bytes | Enhanced supplemental §8 |

Defaults are the graph's values, written once in the settings header
(below) as Phase B's are; a graph varies them, never code.

The scale problem this fixes, stated because it would have sunk the 2022
version here: its reconnection rule is a fixed world distance, 0.1 m by
default with a TODO to make it scale-aware (Params.slang,
nearFieldDistance). The marbles are 1.2 to 2.5 cm, so nearly no pair of
vertices in this scene is 0.1 m apart, and nearly every path would replay
to its end. Enhanced's footprint rule is scaled by the primary ray's
footprint, so it needs no scene tuning (Enhanced §4, Fig. 10).

Departures, each with why:

- **No light tiles.** Enhanced presamples 128 tiles of 1024 lights by power
  for cache coherence (§6.1). The fireflies are chosen uniformly here
  (their power varies only by glow), so tiles would buy coherence only.
  Queued, measured if NEE's loads show up. (Built in Phase 2: the
  packed entries made the initial pass 4.7% faster, so kept;
  2026-10-11-restir-pt-phase2.md.)
- **No dual motion vectors.** They cut disocclusion noise under camera
  motion (§6.4). The marbles' camera is fixed; disocclusions are a moving
  firefly's few pixels. Queued.
- **Triangles become spheres and boxes.** The 2022 reservoir stores a
  reconnection vertex as instance, triangle and barycentrics. This scene
  has spheres and boxes only, so it stores the primitive and a point on it
  in its own coordinates, as contract 4 already stores a light point
  (equal-area octahedral for a sphere; face and two 16-bit coordinates for
  a box), evaluated again where it is read, so a moving shape is placed
  where it is now.
- **ReSTIR DI stays.** Its graphs and passes keep working (additive). The
  PT graph doesn't run them: Enhanced's unified reservoir supersedes them
  (§6.1, supplemental Fig. 4: better and cheaper than DI beside PT).

## What it costs, expected

Enhanced reports 15.5 ms a frame at 1920 × 1080 on an RTX 5880 Ada,
averaged over four scenes (Table 1). Serenity renders 3456 × 2234 (3.7
times the pixels) on an M3 Max, where the path tracer alone takes 30 ms.
Expect ReSTIR PT at full resolution to cost several times the path tracer.
Production runs path-traced reuse at a reduced internal resolution and
upscales; the window already renders at a scale (`--scale`). The frame
time at full and half scale is measured at Phase 1's checkpoint; nothing
is tuned before then.

## How success is judged

The viewer compares two moving images, so the comparison is one frame
each, at the same frame time: ReSTIR PT's frame against the path tracer
taking as many samples per pixel as fit in ReSTIR PT's frame time. Shown
live, side by side, and as frame time plus that one-frame MSE. No
references are rendered until ReSTIR PT is finished.

The gate: if at Phase 1's checkpoint ReSTIR PT does not visibly beat the
path tracer in the glass, the mirrors and the porcelain at equal frame
time, stop and say so before Phase 2.

## The parallel with attention

ReSTIR PT keeps every piece ReSTIR DI mapped (2026-10-10-restir-di-
layout.md, "The parallel with attention"); only what a key is changes. The
2022 code's PathReservoir::add() and merge() are the softmax pick line for
line: a weight w = luminance(F) / p, added to a running sum, the new path
kept when a uniform number times the sum is at most w.

| ReSTIR PT | Attention |
|---|---|
| A candidate path's luminance over its density | exp(score) |
| RIS over the path tree | Sampling one key from softmax over the tree's keys |
| The path reservoir, one path kept and a running weight | Online softmax state |
| Temporal merge, confidence capped at 20 | A KV cache with a window |
| Spatial merge over 3 paired neighbors | Local attention over a neighborhood |
| Duplication maps lowering the cap where one key dominates a region | Down-weighting a key that already dominates nearby queries |
| Vector-valued weights summed for shading | Averaging values with the weights, as attention does, instead of keeping one |

No attention counterpart, stated: the shift (replay and reconnection) and
its Jacobian, and the MIS weights that correct for samples chosen under a
neighbor's scores. They exist because a path is re-expressed for another
pixel; attention's keys never move between queries. The last row is
attention's own rule: a weighted average instead of a sample, used here
only for color.

## The headers

As ReSTIR DI's: the Integrator owns light transport and sees a reservoir
only through its contract; Sample reuse depends on contracts and the
Sampler only; the Pass composes (tools/check_boundaries.sh rule 5).
Corpus IDs are the C++ Core Guidelines' (cpp-guidelines server) and the
performance corpus's (its HTTP client; its MCP server was down).

**`core/contracts/path_reservoir.h`** (new; contract 15, owned by Sample
reuse). The 64-byte path reservoir, Enhanced's Algorithm 1 adapted to
these shapes: W; F as RGB (p̂ = its luminance); the source's seed, the
(pixel, frame) its numbers were drawn from, and the dimension at the
reconnection vertex, so any pass can redraw them (GDSA.3: numbers keyed by
address); M packed in 8 bits beside the path flags (length, reconnection
vertex index, the lobe and NEE-or-BSDF bits at x_{k-1} and x_k); the
reconnection vertex as primitive and point in its own coordinates; its
incoming direction, octahedral 2 × 16 bits; its incoming radiance, RGB;
the cached Jacobian product and NEE light pdf (supplemental §8). Each
narrowing stated where it is packed and decoded where it is loaded
(ES.46; GDSA.18). Related fields as one struct (C.1), static_asserted at
64 bytes, so a gather reads two aligned 32-byte halves (GPU.2).

**`core/passes/restir_pt.h`** (new; Pass). The settings above, each named
once with its source (ES.45), as `restir_di.h` does; a contract for the
shaders in `core/contracts/reuse_settings.h`'s style.

**`core/sampling/pairing.h`, `.cpp`** (new; core). Builds the three reuse
textures on the host, once, seeded: consecutive link indices, n_σ tiled 2
× 2 shuffles (Enhanced eq. 3), each pixel paired with its link's other
end, deltas packed 2 × 8 bits, long links wrapped so the texture tiles
(§3.1). Static data turned once into the layout the shader reads
(CDSA.32). Deterministic, so a test can check its pairing is an
involution and its deltas' deviation is σ.

**`core/frame/schedule.h`** (changed). Pass kinds `restir_pt_initial`,
`restir_pt_temporal`, `restir_pt_spatial`, `restir_pt_shade`,
`restir_pt_duplication`, each refused by name until its phase builds it.

**`metal/integrator/path.metal.h`** (changed). The one path loop
`walk_path<Start, Stop>` gains what it does with each contribution as a
template parameter: the path pass adds it to L as now; the PT initial pass
streams it into the path reservoir with its pdf. One loop, not a copy:
Phase A's copy cost 19% (2026-10-11-path-split-cost.md). Russian roulette
stays the path pass's; the PT pass folds survival into the source pdf
(supplemental §6). Its inputs as one struct (I.4, C.1). The path pass's
image stays the same to the bit, its frame time measured.

**`metal/integrator/nee.metal.h`** (new). NEE at a vertex by RIS over B's
candidates (32 / B²), visibility for the chosen light only, its MIS
weight the M-sample form (supplemental §5). ReSTIR DI's candidate code
(`integrator/restir_di.metal.h`) is what it generalizes.

**`metal/integrator/reconnection.metal.h`** (new). The criteria: the
primary footprint R_pri from x0, x1 and x1's normal; the dual footprint
test (eq. 5, c = 0.02), skipped at a diffuse or emissive x_k; the
roughness test α(x_{k-1}) ≥ 0.2 on the sampled lobe. Pure functions of
two vertices and their lobes (F.8), tested on their own.

**`metal/integrator/shift.metal.h`** (new). The hybrid shift of a stored
path to another pixel's primary hit: replay with the source's numbers
until x_{k-1}, reconnect to the stored x_k, the Jacobian (supplemental eq.
4; 1 for pure replay), the invertibility check, forced NEE reconnection.
Returns the shifted F and Jacobian, or a failure the caller sees: a
failed shift is a value in the interface, not a zero hidden in F (I.1).

**`metal/reuse/path_merge.metal.h`** (new; Sample reuse). The GRIS merge
of shifted paths: pairwise MIS for the spatial merge, Talbot for the
temporal, confidence weights, the cap, the vector-valued weight sum for
shading. It takes F and Jacobians already shifted, so it never calls the
Integrator (rule 5). The streaming pick is `reservoir.metal.h`'s, the one
softmax-pick routine for both ReSTIRs (ES.3).

**`metal/reuse/duplication.metal.h`** (new; Sample reuse). Counts, per
pixel, the reservoirs in its 17 × 17 neighborhood holding the same seed;
D = count / 288; c_cap = lerp(20, 1, D^0.1). One pass a frame.

**`metal/reuse/pairing.metal.h`** (new; Sample reuse). A pixel's paired
neighbor: its delta read from a reuse texture under this frame's flip,
mirror, transpose and offset.

**`metal/passes/restir_pt/`** and pass directories `restir_pt_initial`,
`restir_pt_temporal`, `restir_pt_spatial`, `restir_pt_shade`,
`restir_pt_duplication` (new; Pass). The composition. The spatial pass is
Enhanced's two: a prepass shifting each path to its paired neighbors, then
resampling (§7). Replay-needing pairs compacted with a scan before replay,
the count left on the device (GDSA.10, GDSA.13), so lanes that only
reconnect don't wait on lanes that replay (GPU.4).

**`graphs/restir_pt.toml`** (new). initial → temporal → spatial → shade →
duplication → tone_map.

**`metal/materials/*`** (changed as needed). Lobe-indexed paths need each
lobe's value, pdf and roughness on its own; the coated surface (porcelain)
and the dielectric (glass) have two lobes. Contract 2 already reports the
sampled lobe; what it lacks is added there.

## Phases

Each ends with something to watch, not a reference render.

1. **Initial sampling and spatial reuse, together.** The path tree with
   NEE at every vertex, RIS over it into the path reservoir; then the
   hybrid shift, the criteria, paired neighbors, pairwise MIS, replay
   compaction, color noise. Initial sampling alone reuses nothing and
   shows nothing new, so it is not a stop; its mean is still tested
   against the closed-form floors on the way. The checkpoint: watch it
   live beside the path tracer at equal frame time, at full and half
   scale. The gate above applies here.
2. **Temporal reuse and decorrelation.** Talbot MIS, the cap, the
   duplication map. Watch it live: the caustics and cores are where the
   temporal persistence shows.
3. **Close.** References once, the clip, the write-up.

Measurement at each phase: frame time and the one-frame comparison at
equal time. Anything under about 1%, or not visible, goes to the queue.
