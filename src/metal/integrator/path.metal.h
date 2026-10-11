#pragma once

// Axis: Integrator (the naive path tracer).
//
// The radiance arriving along one camera ray, by the path tracer with next
// event estimation and multiple importance sampling (MIS): pbrt-v4's
// PathIntegrator (section 13.4), its SampleLd() and its emission at a hit.
// At every surface with a lobe that is not delta, a firefly's light is found
// two ways, by a light sample (next event estimation) and by the ray the
// BSDF sends on, striking the firefly, and each is weighted against the
// other by the power heuristic (Veach 1997, chapter 9), as
// logical-overview.md's principle 7 has said since 2026-10-11. Milestone 1's
// baseline, run once per pixel per frame by the path pass
// (metal/passes/path/path.h); its mean over frames converges to the scene's
// full light transport, so it is what every later estimator is measured
// against.
//
// Why two strategies, weighted (decided 2026-10-11, by the user's direction:
// docs/research/2026-10-11-restir-ris-results.md, "Verdict"; measured in
// docs/research/2026-10-11-path-mis.md). A light sample is drawn uniformly
// over the light's cone, so at a near-mirror conductor (GGX alpha down to
// 10^-3) one that lands in the lobe returns f cos L_e / (P p) with f up to
// about 1 / (4 pi alpha^2): unbiased, its variance growing as 1 / alpha^2,
// the rare samples of 10^5 that set the error over thousands of frames. The
// BSDF's density matches such a lobe, and the light's matches a rough
// surface; the power heuristic hands each direction to the strategy whose
// density there is the higher, as production path tracers do (pbrt-v4's
// PathIntegrator, Falcor's). Considered and not taken: the balance
// heuristic (the power heuristic, exponent 2, has the lower variance where
// one density dominates the other, as at a near mirror, Veach 9.2.4, and is
// pbrt-v4's); clamping a sample's radiance (biased, and the reference must
// not be).
//
// The algorithm, whose steps the code below carries by number:
//
//   L = 0, the radiance found; beta = 1, the path's throughput; the ray from
//   the camera; emission_pdf = counted_in_full, how the next surface counts
//   a light it is (step 3), for the camera's own ray; from, the point the
//   ray truly left, the camera's; and the medium the path is in (contract
//   12), no_medium, air, at the camera.
//   For each surface, until Russian roulette ends the path (step 7):
//
//   Step 1  Trace: the nearest surface along the ray; beta *=
//           transmittance(medium, t) (contract 12), what the medium the ray
//           crossed kept of it, t the stretch from the surface the ray truly
//           left, not from the point ray_offset off it the ray starts at: at a
//           marble's scale, and its tint's absorption of some 140 per meter,
//           that 0.1 mm is a percent of its light. 1 in air.
//   Step 2  Escape: if there is none, L += beta x sky(direction), and stop.
//           The sky is not aimed at, so this is the one way it is counted.
//   Step 3  Emission: if the surface is a light (contract 3, light_at),
//           L += beta x L_e x w_b, L_e the emitter's, w_b by emission_pdf:
//             - counted_in_full, for the camera's own ray and a ray a delta
//               lobe sent, which no shadow ray could have reached the light
//               along (glass blocks shadow rays): w_b = 1. Light that glass
//               focuses onto a rough surface is counted so, the one way it
//               can be, until manifold next event estimation takes it over.
//             - a density p_b > 0, for a ray that a lobe that is not delta
//               drew with density p_b, from a surface whose step 5 aimed at
//               lights: w_b = p_b^2 / (p_b^2 + (P p_l)^2), the power
//               heuristic, P = 1 / N the light's chance of being chosen
//               (light_selection/uniform_light.metal.h, light_probability)
//               and p_l the density with which step 5 at `from` draws this
//               direction toward this light (contract 3's light_pdf(): the
//               cone's 1 / its solid angle, and 0 outside it, where w_b is
//               1, since no light sample could have found the light that way).
//             - not_counted, ReSTIR's continuation only (below): nothing.
//           A light's surface scatters what its BSDF scatters: a glowing
//           sphere's nothing, so its path ends at step 6.
//   Step 4  Resolve the surface's BSDF (contract 2).
//   Step 5  Next event estimation, where the BSDF has a lobe that is not
//           delta (aims_at_lights): choose one light uniformly, P = 1 / N
//           (light_selection/uniform_light.metal.h); draw a direction toward
//           it, pdf p_l (contract 3); if it gives no light (L_e = 0, a
//           firefly not yet awake, core/animation/glow.h), stop there, as
//           pbrt-v4's PathIntegrator does (its test of ls->L): the term is 0
//           whatever the shadow ray finds, so the ray is not traced; else
//           trace one shadow ray, to the light's surface, the light itself
//           ignored; if nothing blocks it,
//             L += beta x f(wo, wi) |cos(wi)| x L_e x T / (P x p_l) x w_l,
//           w_l = (P p_l)^2 / ((P p_l)^2 + p_b^2), p_b = pdf(wo, wi), the
//           density with which step 6 here would draw wi (contract 2): the
//           same two densities step 3 weighs a BSDF-drawn ray by, so for any
//           direction both strategies can draw, w_l + w_b = 1, and the
//           light is counted once in expectation. T is the transmittance of
//           the medium the path is in over the shadow ray's length: 1 in
//           air; a light inside the same glass as the surface, dimmed as the
//           glass dims. The term is computed whole before the shadow ray is
//           traced, so across that query the step holds one float3 beside
//           the walk's own state, not f, wi, L_e, T and the densities
//           (GPU.3; ReSTIR's shade does the same,
//           metal/passes/restir_di/restir_di.metal.h). The cost per surface
//           is the same for any number of lights (principle 9).
//   Step 6  Sample the BSDF: wi, f and pdf (contract 2). If pdf is 0, stop.
//           beta *= f |cos(wi)| / pdf. emission_pdf = counted_in_full if
//           the lobe was delta, else pdf: what step 3 at the next surface
//           weighs a light by. If the lobe was a
//           transmission, the path crossed the surface into or out of its
//           shape: arriving from outside, it enters the shape's interior
//           (SurfaceInteraction::interior, contract 1); from inside, it
//           leaves into air. One medium at a time (contract 12): an opaque
//           core inside a tinted marble reflects without leaving the
//           marble's medium, so the stretches between glass and core are
//           dimmed as the glass's. A path in air asks no medium anything;
//           lanes diverge only where a SIMD group straddles glass and air,
//           a marble's few pixels (GPU.4). What a scene with no medium
//           pays for it: docs/research/2026-10-09-medium-cost.md.
//           A transmission also multiplies eta_scale, 1 at the camera, by
//           eta_t^2 (bsdf_eta), undoing the 1 / eta_t^2 its value scaled
//           radiance by (dielectric.metal.h): beta x eta_scale is the
//           throughput as energy sees it, which no boundary changes.
//   Step 7  Russian roulette, from the 4th surface: survive with
//           q = min(the largest channel of beta x eta_scale, 0.95), else
//           stop; beta /= q, so the mean is unchanged. This, not a depth
//           limit, ends paths: every path length keeps a chance, so the
//           mean converges to the full light transport (Veach 1997, 2.4).
//           By beta x eta_scale, not beta, as pbrt-v4's PathIntegrator
//           (etaScale): inside glass of ior 1.5, beta alone is 1 / 2.25 of
//           the energy the path carries, and roulette would end the paths
//           inside a marble twice as often as their light warrants. The
//           mean is the same either way; the noise inside glass is not.
//           Roulette does not depend on the direction drawn, so it enters
//           neither strategy's density.
//   Step 8  Continue: the ray leaves the surface along wi, and from is the
//           surface's point.
//
// What the loop carries between surfaces for MIS: `from`, carried already
// for the medium's stretch, which is the point step 5 there drew its light
// sample from; and emission_pdf, one float in place of the bool ("counts
// emission") it replaces. Nothing else crosses a trace for it: step 3 finds
// p_l at the hit, from `from`, the ray's direction and the light struck.
// So step 5 and step 3 ask their densities of the same point and the same
// direction. A ray starts ray_offset off that point, so near a light's rim
// it can strike the light along a direction the cone from the point itself
// just misses; light_pdf() is 0 there and w_b 1: a sliver of directions no
// light sample draws, counted by the one strategy that does.
//
// emission_pdf's two named values are values of its one purpose, how the
// next surface counts a light (ES.26; ES.45): counted_in_full is 0 and
// not_counted -1, and every density step 6 hands on is above 0 (a sample of
// pdf 0 ends the path), so no density is taken for either. One float and
// two names, rather than a bool beside a float as pbrt-v4 keeps them
// (specularBounce, bsdfPDF), so the loop carries no more across its traces
// than before (GPU.3); that shape was not measured against this one.
//
// The power heuristic, for the strategy that drew a direction with density
// f > 0 beside the other's g >= 0, is f^2 / (f^2 + g^2), in (0, 1]. Pure
// (F.8). Precondition (I.5): both densities below 10^19, so that the sum of
// their squares is a finite float. The BSDF's are at most some 10^8 (GGX at
// least_alpha, toward grazing: metal/materials/resolve.metal.h), and P p_l
// = 1 / (N x the cone's solid angle) reaches 10^19 only for a cone under
// 10^-19 / N sr: a firefly a millimeter across seen from thousands of
// kilometers. Stated here, not tested per sample.
//
// What the camera ray starts from, and what is done with L, are the pass's:
// a point in the pixel drawn anew each frame, and the mean folded into the
// accumulated image (metal/film/accumulate.metal.h).
//
// The numbers a path draws are a function of the pixel, the frame's index
// and the dimension, the count of numbers the path drew before
// (metal/sampler/sampler.metal.h): independent between frames, so their mean
// converges, and any frame can be rendered again exactly, alone (principle 2).
// MIS draws no number: each path draws the numbers it drew before MIS, in
// the same order, and traces the same rays; only what it adds differs.
//
// A safety stop at 256 surfaces bounds one thread's work, against a path
// that roulette keeps alive for long (each survival is at most 0.95 likely,
// and a survivor's throughput is divided by it, so roulette leaves every
// path's expected contribution unchanged). The stop is the one place the
// estimator drops light: the light that reaches the camera only after more
// than 256 surfaces. Every surface that absorbs some of what it scatters
// shrinks that light by its albedo, so it falls off at least as fast as the
// largest albedo to the 256th power: below 1e-18 of the light for albedos of
// 0.85 and under, as the floor's and the brass's are. Only lossless surfaces
// (smooth glass) do not shrink it, and a ray through a glass sphere leaves
// it within a few surfaces. Stated, not hidden; not zero.
//
// Cost per path: one ray to the next surface and at most one shadow ray per
// surface, so about twice the mean path length, a handful of surfaces in
// practice; 512 rays at the safety stop. MIS adds no ray and no read of
// device memory. Per step 5 that reaches its shadow ray it adds one BSDF
// density (contract 2's pdf: a dot product for a Lambert surface; a half
// vector, D and G1 for a conductor; a Fresnel term for a coated one) and
// the heuristic (four multiplies, an add, a divide). Per light struck after
// a lobe that is not delta it adds one light density (the cone from
// `from`: a length, a square root, a cross product, a divide), the light's
// data, transform and glow, read from the constant address space as step
// 3's L_e already reads them, the selection's 1 / N and the heuristic. Each
// is on the lanes that already take step 5 or step 3 (GPU.4). Measured
// against the build before it: docs/research/2026-10-11-path-mis.md.
//
// The split, for ReSTIR DI. The loop above is one function, walk_path(),
// whose start and stop are template parameters, so the reference and ReSTIR
// share every line of light transport but the direct light at the first
// rough surface (logical-overview.md, principle 3; ES.3), each caller
// compiling the loop for its own start and stop:
//
//   radiance()        <camera, at_end>: the loop above, from the camera's
//                     ray to the path's end. The path pass's.
//   camera_segment()  <camera, at_first_aimed>: steps 1 to 4 and, at
//                     surfaces with no lobe that aims at lights, 6 to 8, from
//                     the camera ray to the first surface that aims at lights
//                     (aims_at_lights), stopping there after step 4: its
//                     Shading, and the path's state, L, beta, eta_scale, the
//                     surface's count, and the numbers' counter, which the
//                     caller holds. Or the path's end before any such surface
//                     (it left the scene, reached a light, or roulette or the
//                     safety stop ended it among glass and mirrors), with L.
//                     Every lobe before that surface is delta, so the segment
//                     counts every light it strikes in full, and emission_pdf
//                     is not part of the path state.
//   continue_path()   <after_step_5, at_end>: from that surface, steps 6 to 8
//                     there, then the loop on from the next surface; step 5
//                     there is the caller's.
//
// Why one loop, specialized, and not two loops called in turn: the path
// pass's loop must keep the state it kept before the split live across its
// ray queries, and no more, since that state is what holds its occupancy
// down (docs/research/2026-10-10-path-kernel-counters.md). Two
// loops called in turn compiled the transport twice into the path kernel,
// and the second carried the first surface's Shading across every later ray
// query; one specialized loop compiles once, the same steps over the same
// state, in locals as before the split, so the path pass keeps its time
// (docs/research/2026-10-11-path-split-cost.md, which also records
// the shapes of the same loop that changed the time or, under fast math,
// the rounding).
//
// The path pass calls radiance() alone. Its image changed with MIS, on
// purpose, by the weights alone, and its expectation did not: the one
// change that has broken its bit-identity with the build before it,
// declared (docs/process/MEMORY.md, 2026-10-11) and held instead by
// converging to the same references (docs/research/2026-10-11-path-mis.md).
// ReSTIR's initial pass calls camera_segment() and writes its result as the
// pixel's surface record and path state (contracts/surface_record.h); the
// shade calls continue_path() on what it reads back, so step 5 is skipped
// there and ReSTIR's direct light is the shade's to add
// (metal/passes/restir_di/restir_di.metal.h, step 6). Its numbers continue
// the path's stream from where the segment left it: the three step 5 would
// have drawn are not drawn, which changes which numbers later steps draw,
// not their independence.
//
// ReSTIR's first surface, q, is principle 7's one exception. The shade's
// direct light there is the reservoir's estimate, weighed against no BSDF
// sample (a BSDF candidate, as RTXDI draws one, is ReSTIR's Phase B), so it
// alone counts every firefly's direct light at q. continue_path() therefore
// counts no light that the bounce from q strikes off a lobe that is not
// delta: step 6 at q hands on not_counted in place of its pdf. A light the
// coat's delta lobe reaches from q is counted in full, as everywhere. From
// the next surface on, the continuation is the path tracer's, MIS and all,
// being the same loop. So nothing is counted twice and nothing left out:
// at q each firefly once, by ReSTIR; after q, each light path as the path
// tracer counts it.
//
// What the continuation reads of the surface, beyond its Shading's BSDF, wo
// and medium: the point and geometric normal (leaving it), the flags (which
// side it was arrived at from) and interior (the medium a transmission
// enters). The shade rebuilds those from the record and path state, its
// Shading's BSDF, wo and geometric normal from the record unpacked (the
// geometric normal is the shading normal for every surface so far:
// contracts/surface_record.h), so the continuation leaves from the surface
// ReSTIR weighed, its packing's small difference from the reference's
// stated there; the SurfaceInteraction's material, primitive,
// object_position and shading normal it leaves 0, unread by the
// continuation, and that is a contract of continue_path(): it reads no
// other field of `here.surface`, and adding a read of one adds it to
// contract 5's path state. The path pass never packs.


#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/medium.h"
#include "core/contracts/surface_interaction.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/camera/thin_lens.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/media/media.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/scene/scene_block.metal.h"
#include "metal/shapes/shapes.metal.h"

namespace serenity {
namespace shaders {
namespace path {

constant constexpr uint roulette_from = 3;  // the 4th surface, counting from 0
constant constexpr uint safety_stop = 256;
constant constexpr float max_survival = 0.95f;

// The scene (metal/scene/scene_block.metal.h), its lights chosen among
// uniformly.
using Scene = SceneView<UniformLight>;

// Step 5 at `here`: what one light, chosen uniformly, sends toward wo along
// one direction drawn toward it, f |cos| L_e T / (P p), or 0 where the
// shadow ray to its surface is blocked. T is over `here`'s medium, which the
// shadow ray crosses.
inline float3 next_event(Scene scene, Shading here, thread PathNumbers& numbers) {
    const SelectedLight chosen = select_light(scene.selection(), next_number(numbers));
    const serenity::contracts::LightSample sample =
        sample_light(scene.lights(), chosen.light, to_float3(here.surface.position), next_numbers2(numbers));
    if (sample.pdf <= 0.0f || !metal::any(to_float3(sample.radiance) > 0.0f)) {
        return float3(0.0f);  // no light from it: no shadow ray (step 5)
    }
    const float3 wi = to_float3(sample.direction);
    const float3 f = bsdf_evaluate(here.bsdf, here.wo, wi);
    if (!metal::any(f > 0.0f) ||
        occluded(scene.structure, scene.shapes(), leave(here.surface, wi, sample.distance), sample.primitive)) {
        return float3(0.0f);
    }
    return f * metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) * to_float3(sample.radiance) *
           transmittance(scene.media(), here.medium, sample.distance) / (chosen.probability * sample.pdf);
}

// What the loop carries from one surface to the next (steps 1 to 8): the
// light found, the throughput, eta_scale, whether the next surface counts
// emission, the medium the ray is in, the ray, and where it truly left from
// (the camera, then each surface's point; the ray itself starts ray_offset
// off that surface, so its hit distance is short of the stretch by that
// much, and a medium's stretch is measured from here).
struct Walk {
    float3 L;
    float3 beta;
    float eta_scale;
    bool counts_emission;
    uint medium;
    metal::raytracing::ray along;
    float3 from;
};

// The walk at the camera: nothing found, throughput 1, the camera's own ray,
// which counts emission, in air.
inline Walk walk_from(CameraRay camera) {
    return Walk{float3(0.0f),
                float3(1.0f),
                1.0f,
                true,
                serenity::contracts::no_medium,
                metal::raytracing::ray(camera.origin, camera.direction, 0.0f, unbounded),
                camera.origin};
}

// Where walk_path() starts and where it stops (above, "The split").
enum class Start {
    camera,       // at step 1, along the walk's ray
    after_step_5  // at step 6, at a surface already resolved, whose step 5 is the caller's
};
enum class Stop {
    at_end,         // where the path ends
    at_first_aimed  // also at the first surface that aims at lights, after its step 4
};

// Where walk_path() stopped, and the walk there.
struct Stopped {
    bool at_surface;  // at a surface that aims at lights (Stop::at_first_aimed only); else the path ended
    Shading here;     // that surface, after step 4; read only where at_surface
    uint surface;     // its count along the path, from 0; read only where at_surface
    Walk walk;        // where the path ended, only L is meaningful
};

// Steps 1 to 8 for each surface from `first`, the count of the surface the
// walk starts at, until the path ends or `stop` says; `at` the surface
// Start::after_step_5 starts at, unread by Start::camera.
//
// One loop, compiled for each caller's start and stop: they are template
// parameters, not arguments, so the path pass's walk, <camera, at_end>,
// holds neither (ES.3; GPU.3). Its state is in locals and its steps in its
// own body, steps 6 to 8 a lambda over those locals (F.50, F.52), both
// forced inline (GEN.6, overridden deliberately): so the path pass's loop
// is the loop before the split, keeping the same state live across its
// ray queries (above, "The split").
template <Start start, Stop stop>
[[gnu::always_inline]] inline Stopped walk_path(Scene scene, Walk walk, Shading at, uint first,
                                                thread PathNumbers& numbers) {
    float3 L = walk.L;
    float3 beta = walk.beta;
    float eta_scale = walk.eta_scale;
    bool counts_emission = walk.counts_emission;
    uint medium = walk.medium;
    metal::raytracing::ray along = walk.along;
    float3 from = walk.from;
    const auto carried = [&]() __attribute__((always_inline)) {
        return Walk{L, beta, eta_scale, counts_emission, medium, along, from};
    };

    // Steps 6 to 8 at `here`, the path's surface number `surface`: the walk
    // sent on from it; false where the path ends there.
    const auto scatter = [&](Shading here, uint surface) __attribute__((always_inline)) -> bool {
        // Step 6.
        const serenity::contracts::BsdfSample next = bsdf_sample(here.bsdf, here.wo, next_numbers3(numbers));
        if (next.pdf <= 0.0f) {
            return false;
        }
        const float3 wi = to_float3(next.direction);
        beta *= to_float3(next.value) * metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) / next.pdf;
        counts_emission = (next.lobe & serenity::contracts::lobe_delta) != 0u;
        if ((next.lobe & serenity::contracts::lobe_transmission) != 0u) {
            medium = (here.surface.flags & serenity::contracts::arrived_from_outside) != 0u
                         ? here.surface.interior
                         : serenity::contracts::no_medium;
            const float eta = bsdf_eta(here.bsdf, here.wo);
            eta_scale *= eta * eta;
        }

        // Step 7.
        if (surface >= roulette_from) {
            const float energy = metal::max(metal::max(beta.r, beta.g), beta.b) * eta_scale;
            const float q = metal::min(energy, max_survival);
            if (next_number(numbers) >= q) {
                return false;
            }
            beta /= q;
        }

        // Step 8.
        from = to_float3(here.surface.position);
        along = leave(here.surface, wi);
        return true;
    };

    if constexpr (start == Start::after_step_5) {
        if (!scatter(at, first)) {
            return Stopped{false, Shading{}, 0u, carried()};
        }
    }
    for (uint surface = start == Start::after_step_5 ? first + 1u : first; surface < safety_stop; ++surface) {
        // Step 1.
        const Hit hit = trace(scene.structure, scene.shapes(), along);

        // Step 2.
        if (!hit.found) {
            L += beta * gradient_sky(scene.sky(), along.direction);
            break;
        }
        const float3 point = along.origin + hit.t * along.direction;
        beta *= transmittance(scene.media(), medium, metal::distance(from, point));
        const serenity::contracts::SurfaceInteraction surface_at =
            surface_interaction(scene.shapes(), hit.primitive, point, along.direction);

        // Step 3: a shape's light record is read only where emission counts.
        if (counts_emission) {
            const ShapeLight glowing = light_at(scene.lights(), hit.primitive);
            if (glowing.is_light) {
                L += beta * light_emitted(scene.lights(), glowing.light, point, -along.direction);
            }
        }

        // Step 4.
        const Shading here{surface_at, resolve_bsdf(scene.materials(), scene.textures(), surface_at),
                           -along.direction, medium};

        // Step 5, or the stop before it.
        if constexpr (stop == Stop::at_first_aimed) {
            if (serenity::contracts::aims_at_lights(bsdf_lobes(here.bsdf))) {
                return Stopped{true, here, surface, carried()};
            }
        }
        if (serenity::contracts::aims_at_lights(bsdf_lobes(here.bsdf)) && scene.selection().count > 0u) {
            L += beta * next_event(scene, here, numbers);
        }

        // Steps 6 to 8.
        if (!scatter(here, surface)) {
            break;
        }
    }
    return Stopped{false, Shading{}, 0u, carried()};
}

// The radiance arriving along the camera's ray, by the algorithm above;
// `numbers` the path's own. The path pass's.
inline float3 radiance(Scene scene, CameraRay camera, thread PathNumbers& numbers) {
    return walk_path<Start::camera, Stop::at_end>(scene, walk_from(camera), Shading{}, 0u, numbers).walk.L;
}

// ReSTIR's two parts of the same walk (above, "The split").

// Where the camera segment stopped, and the path's state there.
struct Segment {
    bool at_surface;   // false: the path ended before a surface that aims at lights; only L is meaningful
    Shading here;      // the surface, after step 4
    float3 L;          // the light found on the way
    float3 beta;       // throughput to the surface
    float eta_scale;   // step 6's
    uint surface;      // the surface's count along the path, from 0
};

// The walk from the camera's ray to the first surface that aims at lights,
// stopped there after step 4; or to the path's end before any.
inline Segment camera_segment(Scene scene, CameraRay camera, thread PathNumbers& numbers) {
    const Stopped stopped =
        walk_path<Start::camera, Stop::at_first_aimed>(scene, walk_from(camera), Shading{}, 0u, numbers);
    return Segment{stopped.at_surface, stopped.here,           stopped.walk.L,
                   stopped.walk.beta,  stopped.walk.eta_scale, stopped.surface};
}

// The radiance the path gathers from `segment`'s surface on, L included:
// the walk from step 6 there, step 5 there the caller's. Precondition:
// segment.at_surface. Reads of segment.here.surface only its position,
// geometric normal, flags and interior (above).
inline float3 continue_path(Scene scene, Segment segment, thread PathNumbers& numbers) {
    const Walk walk{segment.L,
                    segment.beta,
                    segment.eta_scale,
                    false,  // unread: step 6 at the segment's surface sets it before the next surface reads it
                    segment.here.medium,
                    metal::raytracing::ray(),  // unread: step 8 at the segment's surface sets it
                    float3(0.0f)};             // unread: step 8 at the segment's surface sets it
    return walk_path<Start::after_step_5, Stop::at_end>(scene, walk, segment.here, segment.surface, numbers).walk.L;
}

}  // namespace path
}  // namespace shaders
}  // namespace serenity
