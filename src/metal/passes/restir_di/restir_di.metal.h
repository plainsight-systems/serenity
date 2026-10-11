#pragma once

// Axis: Pass (ReSTIR DI: the composition its four passes run).
//
// The direct light at each pixel's first rough surface by spatiotemporal
// reservoir resampling, unbiased (Bitterli et al. 2020: Algorithm 5's
// order, Algorithm 6's normalization, visibility reuse): the estimator
// logical-overview.md describes, applied where it applies it, at the first
// surface the camera's path reaches that can be lit by aiming at a light,
// through glass and mirrors (RTXDI's primary surface replacement). Every
// later surface of the path keeps the naive path tracer's next event
// estimation (metal/integrator/path.metal.h), until ReSTIR GI. Its mean
// over frames is the reference's image (principle 3), held by `make
// convergence` on a ReSTIR graph against the reference, as the path tracer
// is.
//
// Where the families meet. This header is the one place that composes
// them: the Integrator's light transport and target (metal/integrator/
// path.metal.h, target.metal.h), Sample reuse's reservoir and merges
// (metal/reuse/), the Light and Light selection families' identities and
// choices (metal/lights/emitter.metal.h, metal/light_selection/), the
// Camera's projection, the Acceleration family's shadow rays and the
// Sampler's numbers. It includes them; none includes it, and none of them
// includes another family's shaders for ReSTIR: Sample reuse asks this
// frame's and last frame's questions of a FrameView this header defines and
// passes (metal/reuse/unbiased.metal.h states what a Frame answers), and the
// Integrator knows no reservoir (change-axes.md: families meet at
// contracts). Only the four passes' shaders include it
// (metal/passes/restir_{initial,temporal,spatial,shade}/), so a change of
// how ReSTIR composes its families changes this file and its passes, and
// no family.
//
// The algorithm, whose steps the code carries by number, by pass
// (core/frame/schedule.h; each pass's header says what its dispatch reads
// and writes):
//
// restir_initial (steps 1 to 3, one kernel: no step needs another pixel's
// result, so nothing is written and read back between them)
//
//   Step 1  Surfaces. The camera's path, as the path tracer runs it
//           (path.metal.h, camera_segment()), to its first surface q with a
//           lobe that is not delta: q's surface record (its resolved Bsdf,
//           point and wo), its path state (throughput, the light found on
//           the way, medium, the numbers' counter), and, in a graph with a
//           temporal merge, where q's point was last frame
//           (contracts/surface_record.h). The path state is written here,
//           before step 2, so it is not alive across step 3's ray query. No
//           such surface: a record whose bsdf is of kind none, and the path
//           state's light is the pixel's.
//   Step 2  Candidates. M times: a light chosen as the naive estimator
//           chooses one, uniformly, probability 1 / N
//           (light_selection/uniform_light.metal.h); a point on it drawn as
//           sample_light() draws a direction (contracts/light_sample.h);
//           streamed into a reservoir with weight p̂_q(y) / p(y), the target
//           (target.metal.h) over the density, both in the light's own area
//           (metal/reuse/reservoir.metal.h, step 2). Finished with Z = M:
//           W = w_sum / (M p̂_q(y)), resampled importance sampling.
//   Step 3  Visibility reuse. One shadow ray from q to the survivor
//           (visible(), below); blocked, W = 0 and M kept. What a pixel then
//           shares with its neighbors and its next frame is a light it can
//           see: the paper's quality gain where lights are occluded, and
//           this scene is marbles. It also opens the invariant every later
//           pass keeps (metal/reuse/unbiased.metal.h): a reservoir holds a
//           sample visible from its surface, or weighs it 0.
//
// restir_temporal
//
//   Step 4  Temporal merge (metal/reuse/temporal.metal.h). Last frame's
//           final reservoir at where q was, if that pixel's surface is
//           similar; its count capped at H x M; Z tested against last
//           frame's surface, lights and structure, and the survivor's
//           visibility from q where it came from last frame. At most one
//           ray.
//
// restir_spatial
//
//   Step 5  Spatial merge (metal/reuse/spatial.metal.h). k neighbors within
//           the radius, each similar or not an input; Z tested against each
//           neighbor that did not produce the survivor, and the survivor's
//           visibility from q where a neighbor produced it. At most k rays.
//
// restir_shade
//
//   Step 6  Shade. The direct light from the final reservoir's sample y,
//           f(wo, wi) |cos θ_x| L_e G x T x W (target.metal.h,
//           direct_light), with no shadow ray: by the invariant, y is
//           visible from q or W is 0, the visibility of the step that kept
//           it. Times the path state's throughput, plus the light it found
//           on the way; then the path continues from q as the naive path
//           tracer does, for the indirect light (path.metal.h,
//           continue_path(), with step 5 at q left out, since this step
//           counted q's direct light: principle 7). The pixel's radiance is
//           folded into the accumulated image, as the path pass folds its
//           own (metal/film/accumulate.metal.h).
//
// Every step after step 1 skips a pixel whose record's bsdf is of kind
// none (its reservoir written empty), and the shade folds in its path
// state's light alone.
//
// The estimators (core/frame/schedule.h): restir_initial and restir_shade
// alone are resampled importance sampling with visibility reuse, and with
// M = 1 naive next event estimation at q; steps 4 and 5 are added by adding
// their passes. Any of them converges to the reference.
//
// Numbers (metal/sampler/sampler.metal.h; GDSA.3). Step 1 and step 6 draw
// the path's own stream, (pixel, frame), from 0, the shade continuing from
// the path state's counter, so the camera's segment draws exactly what the
// path tracer's would. Steps 2 and 3, 4, and 5 each draw their own purpose's
// stream (the purposes below), so no step's numbers repeat another's, and
// each frame's are independent of every other frame's.
//
// What is alive across each ray query (logical-overview.md, "Little is kept
// alive across a ray query"; GPU.3): the camera segment's queries carry the
// path tracer's own state, as the path kernel's do; visibility reuse's
// query carries the record (80 bytes) and the streaming reservoir (32), the
// same as a separate candidates kernel would after reading the record back,
// the path state having been written before it; the merges' queries carry
// their pixel's record and reservoir and the streaming one.
//
// Cost, per pixel per frame, all four passes, k neighbors and M candidates,
// counted from the passes' headers (GDSA.6: passes over memory), in three
// kinds that are not added together:
//
//   intermediate-buffer traffic (the records, path states, previous
//   positions, reservoirs and the accumulated pixel; logical bytes, and DRAM
//   bytes as far as none of it is cached):
//           restir_initial 188 written (80 record, 64 path state, 12
//           previous position, 32 reservoir), nothing read back;
//           restir_temporal 236 read and 32 written; restir_spatial
//           (1 + k) x 112 + 32 when its second walk over the neighbors hits
//           cache, (1 + 2k) x 112 + 32 when it does not; restir_shade 176
//           read (record, path state, reservoir) and the accumulated
//           pixel's 16 read and 16 written. In all, 1368 bytes a pixel at
//           k = 5 with the second walk cached, 1928 with it not. At the
//           display's 3456 x 2234 (P = 7.7 M): 10.6 to 14.9 GB a frame, 26
//           to 37 ms at the M3 Max's 400 GB/s if all of it went to DRAM; at
//           a traced 1728 x 1117, a quarter of that. The surface record is
//           the largest part: the first measured follow-up, if the merges
//           are bound by it, is a smaller record.
//   light data (logical loads, assumed served from cache):
//           every target evaluation evaluates its light point
//           (metal/lights/emitter.metal.h): the 8-byte light record, the
//           16-byte sphere light, its 48-byte transform and 4-byte glow, 76
//           bytes, and a candidate's draw reads the same 76 its target
//           does. M + 2k + 5 evaluations (M in restir_initial, 3 in the
//           temporal merge, at most 1 + 2k in the spatial, 1 in the shade),
//           23 at the settings the graphs are written with: 1748 bytes a
//           pixel of logical loads. They come from arrays of 76 bytes a
//           light in the constant address space, small beside the caches
//           (76 KB for a thousand fireflies), so they are assumed to hit
//           cache, not counted against DRAM; that assumption is the GPU
//           counters' to check (GPU.10). Were none of it cached, it would
//           add 13.5 GB a frame at the display's size.
//   rays    the camera segment's, as the path tracer's; 1 for visibility
//           reuse, at most 1 for the temporal merge, at most k for the
//           spatial merge, none for the shade's direct light: k + 2, 7 at
//           k = 5, against the naive estimator's 1 at q; then the
//           continuation's, as the path tracer's from q on. A ray's
//           traversal of the structure, and the shapes' tests it makes, are
//           its own cost, counted as rays, not as bytes here.
//   numbers 4M; 2; 3k + 1; and the path's own.
//   GPU     4 dispatches, a barrier between each and the next, as each reads
//           what the one before wrote (GPU.8), and the queue barrier before
//           the first, as for any pass that touches what frames share
//           (metal/frame/renderer.h).
//
// The records, path states and reservoirs are intermediates written to
// memory and read back, which GDSA.16 says not to materialize. They are,
// deliberately: the spatial merge reads every pixel's reservoir within its
// radius after every pixel has written it, a grid-wide dependency no tile
// of on-chip memory holds with its halo, and the shade's continuation is
// kept out of the kernels whose queries it would otherwise be alive across
// (GPU.3). Where no such dependency separates steps, they are one kernel:
// steps 1 to 3.
//
// None of it grows with the light count (principle 9): a pixel weighs M
// candidates and k neighbors whatever N is. What grows with N is the noise
// the candidates leave, which the reuse is there to take out, and the
// light arrays' size, which the cache assumption above is about.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/camera.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/reservoir.h"
#include "core/contracts/surface_interaction.h"
#include "core/contracts/surface_record.h"
#include "core/contracts/transform.h"
#include "metal/acceleration/trace.metal.h"
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

// Each step's stream's first dimension (sampler.metal.h, purpose_numbers).
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
// along the record's bsdf.normal (contracts/surface_record.h), ignoring the
// light's own shape. The one visibility test every step makes, so a sample
// is visible from a surface by every step or by none.
inline bool visible(Scene scene, serenity::contracts::SurfaceRecord from, serenity::contracts::LightReceived light);

// Step 2 of temporal reuse, the caller's: where a point is in a frame's
// image, through that frame's camera; none outside it or behind the camera.
struct Reprojected {
    bool found;
    uint2 pixel;
};

// One frame as the merges ask it (metal/reuse/unbiased.metal.h, a Frame):
// its scene, as SceneView makes it of that frame's transforms, glows and
// structure, and its camera framed for the image of `size` pixels. The
// temporal merge is given this frame's and last frame's, which differ in
// data only, so its one ray query serves either (temporal.metal.h, GPU.4).
struct FrameView {
    Scene scene;
    constant serenity::contracts::CameraData* camera;
    uint2 size;

    float target(serenity::contracts::SurfaceRecord q, serenity::contracts::LightPoint y) const;  // target.metal.h
    bool visible(serenity::contracts::SurfaceRecord q, serenity::contracts::LightPoint y) const;  // visible(), above
    float depth(float3 point) const;            // thin_lens.metal.h, camera_depth
    Reprojected project(float3 point) const;    // thin_lens.metal.h, camera_pixel
};

// Steps 2 and 3 at `here`, M = `candidates`; `numbers` the candidates'
// stream. Precondition: here.bsdf.kind is not none and the scene has a
// light.
inline serenity::contracts::Reservoir candidates(Scene scene, serenity::contracts::SurfaceRecord here,
                                                 uint candidates, thread PathNumbers& numbers);

// Step 6: the pixel's radiance, from its record, path state and final
// reservoir; `numbers` the path's stream, from the state's counter.
inline float3 shade(Scene scene, serenity::contracts::SurfaceRecord here, serenity::contracts::PathState state,
                    serenity::contracts::Reservoir final_reservoir, thread PathNumbers& numbers);

}  // namespace restir
}  // namespace shaders
}  // namespace serenity
