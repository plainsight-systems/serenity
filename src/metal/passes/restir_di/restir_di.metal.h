#pragma once

// Axis: Pass (ReSTIR DI: the composition its four passes run).
//
// The direct light at each pixel's first rough surface by spatiotemporal
// reservoir resampling (Bitterli et al. 2020: Algorithm 5's order,
// visibility reuse), configured as NVIDIA's RTXDI ships it (core/passes/
// restir_di.h: its defaults; the user's direction of 2026-10-10, that
// Serenity performs as production software does): the estimator
// logical-overview.md describes, applied where it applies it, at the first
// surface the camera's path reaches that can be lit by aiming at a light,
// through glass and mirrors (RTXDI's primary surface replacement). Every
// later surface of the path keeps the naive path tracer's next event
// estimation (metal/integrator/path.metal.h), until ReSTIR GI.
//
// Two modes (core/passes/restir_di.h, bias_correction; metal/reuse/
// unbiased.metal.h states both exactly). Basic, the default and what ships:
// no shadow ray in the merges, a small darkening on the lit side of shadow
// edges, the shade tracing its survivor's ray or reusing a recent test.
// Raytraced: the paper's unbiased merges, each tracing the rays that test
// visibility in every domain, whose mean over frames is the reference's
// image to within the light point's and the surface record's quantization
// (contracts/light_sample.h, surface_record.h): the check of ReSTIR
// against the reference, by `make convergence` on a ReSTIR graph, as the
// path tracer is checked; not the configuration that ships.
//
// The goal, in two parts, both the implementation's to measure (GPU.10)
// against the naive path graph's baseline (docs/research/
// 2026-10-10-restir-di-layout.md):
//
//   - ReSTIR's added cost per frame in line with production
//     implementations: a few milliseconds at the traced resolution, not a
//     multiple of the naive frame. A ReSTIR frame costs more than a naive
//     frame of the same samples per pixel, by its candidates, its records'
//     and reservoirs' traffic, its visibility ray and its merges; what it
//     adds is counted below.
//   - Much lower error x time than the naive estimator: MSE x seconds
//     against the reference (`make convergence`), the comparison the demo
//     makes. ReSTIR reduces noise; its win is far less error at equal time.
//
// Where the families meet. This header is the one place that composes
// them: the Integrator's light transport and target (metal/integrator/
// path.metal.h, target.metal.h), Sample reuse's reservoir and merges
// (metal/reuse/), the Light and Light selection families' identities and
// choices (metal/lights/emitter.metal.h, metal/light_selection/),
// Material's packed BSDF (through the target's record unpacking), the
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
//           lobe that is not delta: q's surface record (its point, and its
//           wo and Bsdf packed, as a G-buffer holds them), its path state
//           (throughput, the light found on the way, media, the numbers'
//           counter), and, in a graph with a temporal merge, where q's
//           point was last frame (contracts/surface_record.h). The path
//           state is written here, before step 2, so it is not alive across
//           step 3's ray query. No such surface: a record whose kind is
//           none, and the path state's light is the pixel's. Every step
//           asks its questions of the record unpacked (target.metal.h,
//           unpack_record), this one too: the surface the candidates are
//           weighed at is the one the shade shades.
//   Step 2  Candidates. M times: a light chosen as the naive estimator
//           chooses one, uniformly, probability 1 / N
//           (light_selection/uniform_light.metal.h; RTXDI's local light
//           sampling mode, Uniform); a point on it drawn as sample_light()
//           draws a direction, stored quantized and evaluated at its
//           decoded point (contracts/light_sample.h); streamed into a
//           reservoir with weight p̂_q(y) / p(y), the target
//           (target.metal.h) over the density, both in the light's own area
//           (metal/reuse/reservoir.metal.h, step 2). Finished with Z = M:
//           W = w_sum / (M p̂_q(y)), resampled importance sampling.
//   Step 3  Visibility reuse (RTXDI's initial visibility, on by default).
//           One shadow ray from q to the survivor (visible(), below);
//           blocked, W = 0 and M kept (hide()). What a pixel then shares
//           with its neighbors and its next frame is a light it can see:
//           the paper's quality gain where lights are occluded, and this
//           scene is marbles. The reservoir's test is this ray, here and
//           now (contracts/reservoir.h, tested 0).
//
// restir_temporal
//
//   Step 4  Temporal merge (metal/reuse/temporal.metal.h). Last frame's
//           final reservoir where q was: at the reprojected pixel, permuted
//           within its block, or the first similar pixel around it; its
//           count capped at H x M; normalized by the graph's mode, its Z
//           asked of last frame's surface and lights, and in raytraced
//           mode of last frame's structure, with at most one ray.
//
// restir_spatial
//
//   Step 5  Spatial merge (metal/reuse/spatial.metal.h). k neighbors within
//           the radius, each similar in geometry and material or not an
//           input; normalized by the graph's mode, in raytraced mode with
//           at most k rays.
//
// restir_shade
//
//   Step 6  Shade. The direct light from the final reservoir's sample y,
//           where W > 0, and y's visibility V from q:
//
//             - y's test is here and now (its offset and age 0: the
//               initial pass's ray, or a raytraced merge's, from this
//               record this frame): V = 1, no ray. That is every survivor
//               in raytraced mode, by its invariant (unbiased.metal.h),
//               and in a graph with no merge.
//             - in basic mode with visibility_reuse on, y's test within the
//               graph's visibility_age frames and visibility_distance
//               pixels (RTXDI's reuseFinalVisibility, 4 and 16): V = 1, no
//               ray. RTXDI takes a test only from age 1 on, its age 0
//               meaning none is stored; here every sample is tested when
//               it is drawn, so age 0 is a test, and one with offset 0 is
//               this pixel's own.
//             - otherwise: one shadow ray from q (visible(), below). Its
//               result is the reservoir's test, here and now: in a graph
//               with a temporal merge the shade writes it back to the
//               final reservoir, which the next frame's temporal merge
//               reads, with W = 0 where y was hidden (hide()), so the next
//               frame does not carry a light it knows q cannot see.
//
//           Then f(wo, wi) |cos θ_x| L_e G x V x T x W (target.metal.h,
//           direct_light), times the path state's throughput, plus the
//           light it found on the way; then the path continues from q as
//           the naive path tracer does, for the indirect light
//           (path.metal.h, continue_path(), with step 5 at q left out,
//           since this step counted q's direct light: principle 7). The
//           pixel's radiance is folded into the accumulated image, as the
//           path pass folds its own (metal/film/accumulate.metal.h).
//
//           What final-visibility reuse costs in bias, stated: a test made
//           up to visibility_age frames ago, or up to visibility_distance
//           pixels away, is taken for q's now. Every test a reservoir with
//           W > 0 carries found its sample visible (contracts/reservoir.h),
//           so the error is light where q's own ray would have found the
//           sample hidden: within that many pixels of a shadow's edge, and
//           for that many frames behind an occluder or a firefly that
//           moved. A brightening, where basic mode's Z darkens; both at
//           shadow edges, both what production ships, both measured
//           against the reference with the rest of basic mode's error.
//
// Every step after step 1 skips a pixel whose record's kind is none (its
// reservoir written empty), and the shade folds in its path state's light
// alone.
//
// The estimators (core/frame/schedule.h): restir_initial and restir_shade
// alone are resampled importance sampling with visibility reuse, unbiased
// in either mode (no merge, so nothing to normalize), and with M = 1 naive
// next event estimation at q; steps 4 and 5 are added by adding their
// passes. Each converges to the reference in raytraced mode, to within the
// quantization's tolerance; in basic mode to within its bias besides.
//
// Numbers (metal/sampler/sampler.metal.h; GDSA.3). Step 1 and step 6 draw
// the path's own stream, (pixel, frame), from 0, the shade continuing from
// the path state's counter, so the camera's segment draws exactly what the
// path tracer's would. Steps 2 and 3, 4, and 5 each draw their own purpose's
// stream (the purposes below), so no step's numbers repeat another's, and
// each frame's are independent of every other frame's. Step 4's
// permutation draws the permutation purpose's stream at pixel (0, 0): the
// same two numbers in every pixel of the frame.
//
// What is alive across each ray query (logical-overview.md, "Little is kept
// alive across a ray query"; GPU.3): the camera segment's queries carry the
// path tracer's own state, as the path kernel's do; visibility reuse's
// query, the streaming reservoir (28 bytes) and the pixel's index: the
// record and path state were written before step 2, and nothing after the
// query reads the surface, its ray's origin and direction being made
// before it. The raytraced merges' queries carry their pixel's record (32,
// packed) and reservoir and the streaming one. The shade computes the
// unshadowed direct light before its query, so across it what is alive is
// that radiance, the reservoir, the path state and the packed record, which
// the continuation unpacks again after it rather than holding the 72-byte
// unpacked surface across the query: registers traded for a few dozen
// flops, which the counters may reverse (GPU.10).
//
// Cost, per pixel per frame, all four passes, at the settings the graphs
// are written with (M = 8, H = 20, k = 1, basic mode, permutation and
// visibility reuse on), counted from the passes' headers (GDSA.6: passes
// over memory), in kinds that are not added together:
//
//   intermediate-buffer traffic (the records, path states, previous
//   positions, reservoirs and the accumulated pixel; logical bytes, and DRAM
//   bytes as far as none of it is cached):
//           restir_initial 104 written (32 record, 36 path state, 12
//           previous position, 24 reservoir), nothing read back;
//           restir_temporal 124 read and 24 written where the first pixel
//           it tries passes (up to 256 more read at a disocclusion that
//           tries all nine); restir_spatial (1 + k) x 56 + 24 when its
//           second walk over the neighbors hits cache, (1 + 2k) x 56 + 24
//           when it does not, 136 to 192 at k = 1; restir_shade 92 read
//           (record, path state, reservoir), the accumulated pixel's 16
//           read and 16 written, and up to 8 written back to the final
//           reservoir where it traced. In all, 512 to 576 bytes a pixel
//           (4463eee's design: 1368 to 1928). At a traced 1728 x 1117
//           (P = 1.93 M): 0.99 to 1.11 GB a frame, 2.5 to 2.8 ms at the M3
//           Max's 400 GB/s if all of it went to DRAM; at the display's
//           3456 x 2234, four times that. Arithmetic from the counts, not
//           a measurement. In raytraced mode the same, less the shade's
//           write-back.
//   light data (logical loads, assumed served from cache):
//           every target evaluation evaluates its light point
//           (metal/lights/emitter.metal.h): the 8-byte light record, the
//           16-byte sphere light, its 48-byte transform and 4-byte glow, 76
//           bytes, and a candidate's draw reads the same 76 its target
//           does. At most M + 2k + 3 evaluations (M in restir_initial, 2 in
//           the temporal merge, 2k in the spatial, 1 in the shade; each
//           merge's own input weighed by its stored target, contracts/
//           reservoir.h), 13 at the settings above, in either mode: 988
//           bytes a pixel of logical loads. They come from arrays of 76
//           bytes a light in the constant address space, small beside the
//           caches (76 KB for a thousand fireflies), so they are assumed to
//           hit cache, not counted against DRAM; that assumption is the GPU
//           counters' to check (GPU.10).
//   rays    the camera segment's, as the path tracer's; then, against the
//           naive estimator's 1 at q:
//             basic      1 for visibility reuse and at most 1 for the
//                        shade, where it reuses no test: 1 to 2;
//             raytraced  1 for visibility reuse, at most 1 for the temporal
//                        merge and k for the spatial, none for the shade:
//                        k + 2, 3 at k = 1;
//           then the continuation's, as the path tracer's from q on. A
//           ray's traversal of the structure, and the shapes' tests it
//           makes, are its own cost, counted as rays, not as bytes here.
//   numbers 4M; 2, and the frame's 2 for the permutation; 3k + 1; and the
//           path's own.
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
#include "core/passes/restir_di.h"
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
constant constexpr uint candidates_numbers = 1u << 24;   // steps 2 and 3
constant constexpr uint temporal_numbers = 2u << 24;     // step 4
constant constexpr uint spatial_numbers = 3u << 24;      // step 5
constant constexpr uint permutation_numbers = 4u << 24;  // step 4's permutation, drawn at pixel (0, 0)

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
// along the surface's unpacked shading normal (contracts/surface_record.h),
// ignoring the light's own shape. The one visibility test every step
// makes, so a sample is visible from a surface by every step or by none,
// and a test one step made is the one another would have made.
inline bool visible(Scene scene, RecordedSurface from, serenity::contracts::LightReceived light);

// Step 2 of temporal reuse, the caller's: where a point is in a frame's
// image, through that frame's camera, rounded to the nearest pixel; none
// outside it or behind the camera.
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

    // target.metal.h, unpacking q (unpack_record) for the evaluation
    float target(serenity::contracts::SurfaceRecord q, serenity::contracts::LightPoint y) const;
    // visible(), above; asked only in raytraced mode
    bool visible(serenity::contracts::SurfaceRecord q, serenity::contracts::LightPoint y) const;
    float depth(float3 point) const;          // thin_lens.metal.h, camera_depth
    Reprojected project(float3 point) const;  // thin_lens.metal.h, camera_pixel
};

// Step 4's permutation for the frame of `frame_numbers` (the permutation
// purpose's stream at pixel (0, 0)): two numbers in [0, 3], x's and y's;
// the identity when `settings` turn permutation off.
inline BlockPermutation frame_permutation(PathNumbers frame_numbers, serenity::passes::RestirDi settings);

// Steps 2 and 3 at `here`, the pixel's record unpacked, M = `candidates`;
// `numbers` the candidates' stream. Precondition: here's kind is not none
// and the scene has a light.
inline serenity::contracts::Reservoir candidates(Scene scene, RecordedSurface here, uint candidates,
                                                 thread PathNumbers& numbers);

// Step 6's result: the pixel's radiance, and whether the shade traced its
// survivor's ray, in which case `reservoir`, carrying that test (and W = 0
// where the sample was hidden), is written back for the next frame.
struct Shaded {
    float3 radiance;
    bool traced;
    serenity::contracts::Reservoir reservoir;  // the final reservoir, its test the shade's where traced
};

// Step 6, from the pixel's record, path state and final reservoir, by the
// graph's `settings` (its mode and visibility reuse); `numbers` the path's
// stream, from the state's counter.
inline Shaded shade(Scene scene, serenity::contracts::SurfaceRecord here, serenity::contracts::PathState state,
                    serenity::contracts::Reservoir final_reservoir, serenity::passes::RestirDi settings,
                    thread PathNumbers& numbers);

}  // namespace restir
}  // namespace shaders
}  // namespace serenity
