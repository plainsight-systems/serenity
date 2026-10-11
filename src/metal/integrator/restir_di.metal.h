#pragma once

// Axis: Integrator (ReSTIR DI).
//
// The direct light at each pixel's first rough surface by spatiotemporal
// reservoir resampling, unbiased (Bitterli et al. 2020: Algorithm 5's
// order, Algorithm 6's normalization, visibility reuse): the estimator
// logical-overview.md describes, applied where it applies it, at the first
// surface the camera's path reaches that can be lit by aiming at a light,
// through glass and mirrors (RTXDI's primary surface replacement). Every
// later surface of the path keeps the naive path tracer's next event
// estimation (path.metal.h), until ReSTIR GI. Its mean over frames is the
// reference's image (principle 3), held by `make convergence` on a ReSTIR
// graph against the reference, as the path tracer is.
//
// The algorithm, whose steps the code carries by number, one or two steps a
// pass (core/frame/schedule.h; each pass's header says what its dispatch
// reads and writes):
//
//   Step 1  Surfaces (passes/surfaces). The camera's path, as the path
//           tracer runs it (path.metal.h, camera_segment()), to its first
//           surface q with a lobe that is not delta: q's surface record
//           (its point, wo and resolved BSDF), its path state (throughput,
//           the light found on the way, medium, the numbers' counter), and,
//           in a graph with a temporal merge, where q's point was last
//           frame (contracts/surface_record.h). No such surface: a record of
//           kind none, and the path state's light is the pixel's.
//   Step 2  Candidates (passes/restir_candidates). M times: a light chosen
//           as the naive estimator chooses one, uniformly, probability 1 /
//           N (light_selection/uniform_light.metal.h); a point on it drawn
//           as sample_light() draws a direction (contracts/light_sample.h);
//           streamed into a reservoir with weight p̂_q(y) / p(y), the target
//           (target.metal.h) over the density, both in the light's own
//           area (metal/reuse/reservoir.metal.h, step 2). Finished with Z =
//           M: W = w_sum / (M p̂_q(y)), resampled importance sampling.
//   Step 3  Visibility reuse (passes/restir_candidates). One shadow ray from
//           q to the survivor; blocked, W = 0 and M kept. What a pixel then
//           shares with its neighbors and its next frame is a light it can
//           see: the paper's quality gain where lights are occluded, and
//           this scene is marbles. It also opens the invariant every later
//           pass keeps (metal/reuse/unbiased.metal.h): a reservoir holds a
//           sample visible from its surface, or weighs it 0.
//   Step 4  Temporal merge (passes/restir_temporal;
//           metal/reuse/temporal.metal.h). Last frame's final reservoir at
//           where q was, if that pixel's surface is similar; its count
//           capped at H x M; Z tested against last frame's surface, lights
//           and structure, and the survivor's visibility from q where it
//           came from last frame. At most one ray.
//   Step 5  Spatial merge (passes/restir_spatial;
//           metal/reuse/spatial.metal.h). k neighbors within the radius,
//           each similar or not an input; Z tested against each neighbor
//           that did not produce the survivor, and the survivor's
//           visibility from q where a neighbor produced it. At most k rays.
//   Step 6  Shade (passes/restir_shade). The direct light from the final
//           reservoir's sample y, f(wo, wi) |cos θ_x| L_e G x T x W
//           (target.metal.h, direct_light), with no shadow ray: by the
//           invariant, y is visible from q or W is 0, the visibility of the
//           merge that produced it. Times the path state's throughput, plus
//           the light it found on the way; then the path continues from q as
//           the naive path tracer does, for the indirect light
//           (path.metal.h, continue_path(), with step 5 at q left out, since
//           this step counted q's direct light: principle 7). The pixel's
//           radiance is folded into the accumulated image, as the path pass
//           folds its own (metal/film/accumulate.metal.h).
//
// Every pass after step 1 skips a pixel whose record is of kind none (its
// reservoir written empty), and the shade folds in its path state's light
// alone.
//
// The estimators (core/frame/schedule.h): steps 1, 2, 3 and 6 alone are
// resampled importance sampling with visibility reuse, and with M = 1 naive
// next event estimation at q traced one pass earlier; steps 4 and 5 are
// added by adding their passes. Any of them converges to the reference.
//
// Numbers (metal/sampler/sampler.metal.h; GDSA.3). Step 1 and step 6 draw
// the path's own stream, (pixel, frame), from 0, the shade continuing from
// the path state's counter, so the camera's segment draws exactly what the
// path tracer's would. Steps 2 and 3, 4, and 5 each draw their own purpose's
// stream (the purposes below), so no pass's numbers repeat another's, and
// each frame's are independent of every other frame's.
//
// Cost, per pixel per frame, all five passes, k neighbors and M candidates,
// counted from the passes' headers (GDSA.6: passes over memory):
//
//   bytes   surfaces 140 written (64 record, 64 path state, 12 previous
//           position); candidates 64 + 16; temporal 172 + 16; spatial
//           (1 + 2k) x 80 + 16 at most, (1 + k) x 80 + 16 when its second
//           walk over the neighbors hits cache; shade 64 + 64 + 16 read and
//           the accumulated pixel's 16 read and 16 written. In all, 1080
//           bytes a pixel at k = 5 with the second walk cached, 1480 with
//           it not. At the display's 3456 x 2234 (P = 7.7 M): 8.3 to 11.4
//           GB a frame, 21 to 29 ms at the M3 Max's 400 GB/s were none of
//           it cached; at a traced 1728 x 1117, a quarter of that. The
//           surface record is two thirds of it: the first measured
//           follow-up, if the passes are bound by it, packs the BSDF.
//   rays    the camera segment's, as the path tracer's; 1 for visibility
//           reuse, at most 1 for the temporal merge, at most k for the
//           spatial merge, none for the shade's direct light: k + 2, 7 at
//           k = 5, against the naive estimator's 1 at q; then the
//           continuation's, as the path tracer's from q on.
//   target  M in the candidates, 3 in the temporal merge, at most 1 + 2k in
//           the spatial merge, 1 in the shade: M + 2k + 5, 23 at the
//           settings the graphs are written with.
//   numbers 4M; 2; 3k + 1; and the path's own.
//   GPU     5 dispatches, a barrier between each and the next, as each reads
//           what the one before wrote (GPU.8), and the queue barrier before
//           the first, as for any pass that touches what frames share
//           (metal/frame/renderer.h).
//
// The records, path states and reservoirs are intermediates written to
// memory and read back, which GDSA.16 says not to materialize. They are,
// deliberately: the spatial merge reads every pixel's reservoir within 30
// pixels after every pixel has written it, a grid-wide dependency no tile
// of on-chip memory holds with its halo, and the split into passes is what
// keeps each kernel's live state across its ray queries small
// (logical-overview.md, "Little is kept alive across a ray query"; GPU.3).
// Whether surfaces and candidates, which share no neighbor, run better as
// one kernel is a question for the counters, as every pass's time is.
//
// None of it grows with the light count (principle 9): a pixel weighs M
// candidates and k neighbors whatever N is. What grows with N is the noise
// the candidates leave, which the reuse is there to take out.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/reuse_constants.h"
#include "core/contracts/surface_interaction.h"
#include "core/contracts/surface_record.h"
#include "core/contracts/transform.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/integrator/path.metal.h"
#include "metal/integrator/target.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/reuse/spatial.metal.h"
#include "metal/reuse/temporal.metal.h"
#include "metal/reuse/unbiased.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/scene/scene_block.metal.h"

namespace serenity {
namespace shaders {
namespace restir {

// The scene, its lights chosen among uniformly, as the path tracer's.
using Scene = SceneView<UniformLight>;

// Each pass's stream's first dimension (sampler.metal.h, purpose_numbers).
constant constexpr uint candidates_numbers = 1u << 24;  // steps 2 and 3
constant constexpr uint temporal_numbers = 2u << 24;    // step 4
constant constexpr uint spatial_numbers = 3u << 24;     // step 5

// Step 1's result for one pixel.
struct Surfaces {
    serenity::contracts::SurfaceRecord record;
    serenity::contracts::PathState state;
    float3 previous_position;  // where the record's point was last frame; unset without last_transforms
};

// Step 1: the camera's segment along `ray`, as a record and a path state;
// `numbers` the path's stream, which the state's counter records.
// `last_transforms`, last frame's shape transforms in a graph with a
// temporal merge and null otherwise, places the surface's point as last
// frame placed its shape (previous_position()).
inline Surfaces first_surface(Scene scene, CameraRay ray, constant serenity::contracts::Transform* last_transforms,
                              thread PathNumbers& numbers);

// Where `surface`'s point was last frame: its object-space point placed by
// its shape's transform among `last_transforms`, last frame's.
inline float3 previous_position(constant serenity::contracts::Transform* last_transforms,
                                serenity::contracts::SurfaceInteraction surface);

// Whether `light`, as evaluate_light_point() gave it to `from`'s point, is
// visible from `from`: one shadow ray against `scene`'s structure, leaving
// along the record's normal (contracts/surface_record.h), ignoring the
// light's own shape. The one visibility test every step makes, so a sample
// is visible from a surface by every step or by none.
inline bool visible(Scene scene, serenity::contracts::SurfaceRecord from, serenity::contracts::LightReceived light);

// A surface as one frame saw it, for Z's test (metal/reuse/unbiased.metal.h,
// supports()): `scene` that frame's, `record` the surface's.
struct SeenFrom {
    Scene scene;
    serenity::contracts::SurfaceRecord record;

    float target(serenity::contracts::LightPoint y) const;
    bool visible(serenity::contracts::LightPoint y) const;
};

// Steps 2 and 3 at `here`, M = `candidates`; `numbers` the candidates'
// stream. Precondition: here.kind is not none and the scene has a light.
inline serenity::contracts::Reservoir candidates(Scene scene, serenity::contracts::SurfaceRecord here,
                                                 uint candidates, thread PathNumbers& numbers);

// Step 6: the pixel's radiance, from its record, path state and final
// reservoir; `numbers` the path's stream, from the state's counter.
inline float3 shade(Scene scene, serenity::contracts::SurfaceRecord here, serenity::contracts::PathState state,
                    serenity::contracts::Reservoir final_reservoir, thread PathNumbers& numbers);

}  // namespace restir
}  // namespace shaders
}  // namespace serenity
