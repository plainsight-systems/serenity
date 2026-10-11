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
// Two modes (core/contracts/reuse_settings.h, bias_correction; metal/reuse/
// unbiased.metal.h states both exactly). Basic, the default and what ships:
// no shadow ray in the merges, a darkening on the lit side of shadow edges,
// at its worst a just-revealed light at M / (M + C) of its light for a
// frame (unbiased.metal.h), the shade tracing its survivor's ray or reusing
// a recent test.
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
// Where the families meet. This header composes them and decides nothing
// they decide (decided 2026-10-11, by the review of f86e7ae, which found
// the first surface, the candidates' transport and the shade's in this
// file):
//
//   Integrator    metal/integrator/restir_di.metal.h: the light transport,
//                 the first surface (first_surface()), one candidate drawn
//                 and weighed (draw_candidate(), through target.metal.h),
//                 visibility (visible()), and the shade's transport
//                 (unshadowed(), continue_from()). Sees a reservoir only as
//                 contract 14's data; includes no Sample reuse and no Pass.
//   Sample reuse  metal/reuse/: streaming and finishing a reservoir
//                 (reservoir.metal.h), the merges and their normalization
//                 (temporal, spatial, similar, unbiased), and the rule for
//                 taking a stored test (visibility.metal.h). Depends on
//                 contracts and the Sampler; includes no Integrator and no
//                 Pass. It asks this frame's and last frame's questions of a
//                 FrameView this header defines and passes
//                 (metal/reuse/unbiased.metal.h states what a Frame answers).
//   Pass          this header and the four passes' kernels
//                 (metal/passes/restir_{initial,temporal,spatial,shade}/):
//                 the order of the steps, the numbers each draws, which
//                 family each call goes to, and the bindings and barriers.
//                 Includes both families; neither includes it.
//
// So a dependency points from the Pass to the Integrator and to Sample
// reuse, and from each family to contracts, never between the two
// families (change-axes.md: families meet at contracts); tools/
// check_boundaries.sh holds the two families to it. A change of how ReSTIR
// composes its families changes this file and its passes, and no family; a
// change of its light transport changes the Integrator alone, and of its
// reuse Sample reuse alone.
//
// The algorithm, whose steps the code carries by number, by pass
// (core/frame/schedule.h; each pass's header says what its dispatch reads
// and writes):
//
// restir_initial (steps 1 to 3, one kernel: no step needs another pixel's
// result, so nothing is written and read back between them)
//
//   Step 1  Surfaces (the Integrator's first_surface()). The camera's
//           path, as the path tracer runs it (path.metal.h,
//           camera_segment()), to its first surface q with a
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
//   Step 2  Candidates. M times, the Integrator's draw_candidate(): a
//           light chosen as the naive estimator chooses one, uniformly,
//           probability 1 / N (light_selection/uniform_light.metal.h;
//           RTXDI's local light sampling mode, Uniform), placed once; a
//           point on it drawn as sample_light() draws a direction, stored
//           quantized and evaluated at its decoded point
//           (contracts/light_sample.h); its weight p̂_q(y) / p(y), the
//           target (target.metal.h) over the density, both in the light's
//           own area. Then Sample reuse streams it into a reservoir
//           (metal/reuse/reservoir.metal.h, step 2), and finishes with pi =
//           1 and piSum = M: W = w_sum / (M p̂_q(y)), resampled importance
//           sampling.
//   Step 3  Visibility reuse (RTXDI's initial visibility, on by default).
//           One shadow ray from q to the survivor (the Integrator's
//           visible()); blocked, W = 0 and M kept (Sample reuse's hide(),
//           metal/reuse/unbiased.metal.h). What a pixel then shares
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
//           count capped at H times this pixel's; normalized by
//           RTXDI's MIS-like weights (pi over piSum, metal/reuse/
//           unbiased.metal.h), asked of last frame's surface and lights,
//           and in raytraced mode of last frame's structure, with at most
//           one ray.
//
// restir_spatial
//
//   Step 5  Spatial merge (metal/reuse/spatial.metal.h). k neighbors within
//           the radius, each similar in geometry and material or not an
//           input; normalized by the same weights, in raytraced mode with
//           at most k rays.
//
// restir_shade
//
//   Step 6  Shade. The direct light from the final reservoir's
//           sample y, where W > 0, without visibility first (the
//           Integrator's unshadowed(): f(wo, wi) |cos θ_x| L_e G x T x W,
//           target.metal.h, direct_light); then y's visibility V from q,
//           where that light is not 0:
//
//             - from the reservoir's own test, where Sample reuse's rule
//               takes it (metal/reuse/visibility.metal.h, takes_test(),
//               which states the rule and the bias it costs): a test here
//               and now always, which is every survivor in raytraced mode,
//               by its invariant (unbiased.metal.h), and every survivor in
//               a graph with no merge; in basic mode with visibility_reuse
//               on, a recent and near one (RTXDI's reuseFinalVisibility).
//               V = 1, no ray.
//             - otherwise: one shadow ray from q (the Integrator's
//               visible()). Its result is the reservoir's test, here and
//               now (tested_here()): in a graph with a temporal merge the
//               shade writes it back to the final reservoir, which the
//               next frame's temporal merge reads, with W = 0 where y was
//               hidden (hide()), so the next frame does not carry a light
//               it knows q cannot see.
//
//           Then that light times V, times the path state's throughput,
//           plus the light it found on the way; then the path continues
//           from q as the naive path tracer does, for the indirect light
//           (the Integrator's continue_from(): path.metal.h,
//           continue_path(), with step 5 at q left out, since this step
//           counted q's direct light, and with no firefly counted that the
//           bounce from q strikes off a lobe that is not delta; past q, the
//           path tracer's MIS: principle 7). The pixel's radiance is
//           folded into the accumulated image, as the path pass folds its
//           own (metal/film/accumulate.metal.h).
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
// that radiance, the reservoir, the path state and the packed record (32
// bytes), which the continuation unpacks again after it
// (metal/integrator/restir_di.metal.h, continue_from()) rather than the
// shade holding the 72-byte unpacked surface across the query: registers
// traded for two unit-vector decodes. The query is in the shade's code in
// every graph, the RIS graph's included, though there no pixel traces it
// (every survivor's test is the initial pass's, here and now:
// takes_test()); registers are allocated across it all the same. Unpacking
// once, which the review of f86e7ae asked for, was measured slower in the
// RIS graph, and is set aside (decided 2026-10-11:
// docs/research/2026-10-10-restir-di-layout.md, "The review of f86e7ae";
// GPU.3, GPU.10). The merges' graphs, where the shade traces, are to
// measure the trade again.
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
//           reservoir where it traced. In all, 568 to 576 logical bytes a
//           pixel, the spatial merge's second walk counted whether or not it
//           hits cache (a hit moves it from DRAM to cache traffic, not to
//           nothing): 512 to 576 of them DRAM traffic, the first figure
//           when the second walk hits. At a traced 1728 x 1117 (P = 1.93
//           M): 1.10 to 1.11 GB a frame of logical traffic. At the M3
//           Max's peak 400 GB/s that takes at least 2.8 ms, were all of it
//           DRAM traffic: an ideal-bandwidth floor on the time, not a bound
//           above it, since caches, access patterns, occupancy, traversal
//           and arithmetic only add to it (decided 2026-10-11, by the
//           review of f86e7ae, which found it written as "at most"). At
//           the display's 3456 x 2234, four times that. Arithmetic from the
//           counts, not a measurement: which bytes reach DRAM is the GPU
//           counters' to say, and the time the timings' (GPU.10). In
//           raytraced mode the same, less the shade's write-back.
//   light data (logical loads at the source, assumed served from cache):
//           each light a pixel asks about is placed once
//           (metal/lights/emitter.metal.h, place_light(): the 8-byte light
//           record, the 16-byte sphere light, its 48-byte transform and its
//           4-byte glow, 76 bytes), and every question asked of it reads
//           that one value: a candidate's draw, its evaluation and its
//           density are one placement (decided 2026-10-11, by the review of
//           f86e7ae: each had read the light's data again, and the draw its
//           transform twice, 332 bytes a candidate at the source). Each
//           other target evaluation is one placement: at most 2 in the
//           temporal merge and 2k in the spatial (each merge's own input
//           weighed by its stored target, and its source's pi read from
//           the source's, contracts/reservoir.h; in a merge, one evaluation
//           to stream each other input and one for piSum at each input
//           that is neither the source nor the pixel's own:
//           metal/reuse/unbiased.metal.h), and 1 in the shade.
//           Placements: M + 2k + 3, 13 at the settings above in either
//           mode, (M + 2k + 3) x 76 = 988 bytes a pixel; in a graph with no
//           merge (graphs/restir_ris.toml), M + 1, 684 bytes. Counted at
//           the source: what the compiled code loads is the counters' to
//           show (GPU.10). They come from arrays of 76 bytes a light in
//           the constant address space, small beside the caches (76 KB for
//           a thousand fireflies), so they are assumed to hit cache, not
//           counted against DRAM; that assumption is the GPU counters' to
//           check (GPU.10).
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
#include "core/contracts/reuse_settings.h"
#include "core/contracts/surface_record.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/integrator/restir_di.metal.h"
#include "metal/integrator/target.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/reuse/reservoir.metal.h"
#include "metal/reuse/spatial.metal.h"
#include "metal/reuse/temporal.metal.h"
#include "metal/reuse/unbiased.metal.h"
#include "metal/reuse/visibility.metal.h"
#include "metal/sampler/sampler.metal.h"

namespace serenity {
namespace shaders {
namespace restir {

// Each step's stream's first dimension (sampler.metal.h, purpose_numbers).
constant constexpr uint candidates_numbers = 1u << 24;   // steps 2 and 3
constant constexpr uint temporal_numbers = 2u << 24;     // step 4
constant constexpr uint spatial_numbers = 3u << 24;      // step 5
constant constexpr uint permutation_numbers = 4u << 24;  // step 4's permutation, drawn at pixel (0, 0)

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
// Declared, and defined with the merges that ask it (restir_temporal,
// restir_spatial), which the Metal backend refuses by name until then
// (metal/frame/renderer.h).
struct FrameView {
    Scene scene;
    constant serenity::contracts::CameraData* camera;
    uint2 size;

    // the Integrator's target (target.metal.h), unpacking q (unpack_record)
    float target(serenity::contracts::SurfaceRecord q, serenity::contracts::LightPoint y) const;
    // the Integrator's visible(); asked only in raytraced mode
    bool visible(serenity::contracts::SurfaceRecord q, serenity::contracts::LightPoint y) const;
    float depth(float3 point) const;          // thin_lens.metal.h, camera_depth
    Reprojected project(float3 point) const;  // thin_lens.metal.h, camera_pixel
};

// Step 4's permutation for the frame of `frame_numbers` (the permutation
// purpose's stream at pixel (0, 0)): two numbers in [0, 3], x's and y's;
// the identity when `settings` turn permutation off. Declared, and defined
// with the temporal merge, as FrameView is.
inline BlockPermutation frame_permutation(PathNumbers frame_numbers, serenity::contracts::ReuseSettings settings);

// Steps 2 and 3 at `here`, the pixel's record unpacked, M = `count`;
// `numbers` the candidates' stream. Precondition: here's kind is not none
// and the scene has a light.
//
// Each candidate draws four numbers whether or not it yields a sample (a
// light, a point on it, the reservoir's choice), in that order, and every
// candidate counts in M, one drawn from inside its light included: it is a
// draw whose sample does not exist, weighing 0, as the naive estimator's
// next event counts nothing for it (GDSA.3: the counter a function of the
// candidate's position in the stream, never of what an earlier one drew).
// The survivor's answer from its light, the direction, distance and shape
// its shadow ray needs, is kept beside the stream as it is kept, so step 3
// evaluates nothing again; across step 3's query what is alive is the
// stored reservoir and that answer (above; GPU.3).
inline serenity::contracts::Reservoir candidates(Scene scene, RecordedSurface here, uint count,
                                                 thread PathNumbers& numbers) {
    const Lights lights = scene.lights();
    const UniformLight selection = scene.selection();
    Streaming reservoir = empty_reservoir();
    serenity::contracts::LightReceived survivor{};
    for (uint i = 0; i < count; ++i) {
        // Step 2: the Integrator draws and weighs, Sample reuse streams.
        const float choose = next_number(numbers);
        const float2 point_numbers = next_numbers2(numbers);
        const float u = next_number(numbers);
        const Weighed weighed = draw_candidate(lights, selection, here, choose, point_numbers);
        if (stream_candidate(reservoir, Candidate{weighed.sample, weighed.target, weighed.weight}, u)) {
            survivor = weighed.light;
        }
    }
    // Resampled importance sampling's normalization: pi = 1, piSum = M.
    const serenity::contracts::Reservoir chosen = finish(reservoir, Normalization{1.0f, float(reservoir.count)});

    // Step 3: visibility reuse, its test here and now (tested 0, as finish()
    // left it).
    if (chosen.weight > 0.0f && !visible(scene, here, survivor)) {
        return hide(chosen);
    }
    return chosen;
}

// Step 6's result: the pixel's radiance, and whether the shade traced its
// survivor's ray, in which case `reservoir`, carrying that test (and W = 0
// where the sample was hidden), is what the next frame's temporal merge is
// to read. Writing it back is the temporal merge's to add with it
// (restir_shade.h): no graph this backend runs has one yet, so nothing
// would read it, and the shade's kernel writes nothing back.
struct Shaded {
    float3 radiance;
    bool traced;
    serenity::contracts::Reservoir reservoir;  // the final reservoir, its test the shade's where traced
};

// Step 6, from the pixel's record, path state and final reservoir, by the
// graph's `settings` (its mode and visibility reuse); `numbers` the path's
// stream, from the state's counter. The record is unpacked for the direct
// light, and by continue_from() again for the continuation, so the
// unpacked surface is not alive across the visibility query (above).
inline Shaded shade(Scene scene, serenity::contracts::SurfaceRecord record, serenity::contracts::PathState state,
                    serenity::contracts::Reservoir final_reservoir, serenity::contracts::ReuseSettings settings,
                    thread PathNumbers& numbers) {
    if (packed_kind(record.bsdf) == serenity::contracts::BsdfKind::none) {  // no surface (first_surface())
        return Shaded{to_float3(state.radiance), false, final_reservoir};
    }
    Shaded shaded{float3(0.0f), false, final_reservoir};
    float3 direct = float3(0.0f);
    if (serenity::contracts::holds_light(final_reservoir.sample) && final_reservoir.weight > 0.0f) {
        const RecordedSurface here = unpack_record(record);
        const Unshadowed lit =
            unshadowed(scene, here, unpack_medium(state.media), final_reservoir.sample, final_reservoir.weight);
        direct = lit.radiance;
        // V: nothing to see, or seen already (takes_test()), or one ray.
        if (metal::any(direct > 0.0f) && !takes_test(final_reservoir.tested, settings)) {
            shaded.traced = true;
            shaded.reservoir = tested_here(final_reservoir);
            if (!visible(scene, here, lit.light)) {
                direct = float3(0.0f);
                shaded.reservoir = hide(shaded.reservoir);
            }
        }
    }
    shaded.radiance = continue_from(scene, record, state, direct, numbers);
    return shaded;
}

}  // namespace restir
}  // namespace shaders
}  // namespace serenity
