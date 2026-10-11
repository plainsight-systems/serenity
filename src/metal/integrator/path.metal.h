#pragma once

// Axis: Integrator (the naive path tracer).
//
// The radiance arriving along one camera ray, by the textbook path tracer
// with next event estimation: pbrt-v4's SimplePathIntegrator (section 13.3),
// which counts each light path one way and weighs no two strategies against
// each other, as logical-overview.md's principle 7 requires of every
// estimator here. Milestone 1's baseline, run once per pixel per frame by the
// path pass (metal/passes/path/path.h); its mean over frames converges to
// the scene's full light transport, so it is what every later estimator is
// measured against.
//
// The algorithm, whose steps the code below carries by number:
//
//   L = 0, the radiance found; beta = 1, the path's throughput; the ray from
//   the camera; "counts emission", true for the camera's own ray; and the
//   medium the path is in (contract 12), no_medium, air, at the camera.
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
//   Step 3  Emission: if the surface is a light (contract 3, light_at) and
//           the path counts emission, L += beta x L_e, L_e the emitter's. A
//           light's surface scatters what its BSDF scatters: a glowing
//           sphere's nothing, so its path ends at step 6. The path counts
//           emission on the camera's own ray and after a delta lobe, which
//           no shadow ray could have reached the light through (glass
//           blocks shadow rays). After any other bounce the light was
//           counted at the surface before, by step 5, so it adds nothing
//           here (principle 7). Light that glass focuses onto a rough surface
//           is therefore counted here, the one way it can be, until manifold
//           next event estimation takes it over.
//   Step 4  Resolve the surface's BSDF (contract 2).
//   Step 5  Next event estimation, where the BSDF has a lobe that is not
//           delta (aims_at_lights): choose one light uniformly, P = 1 / N
//           (light_selection/uniform_light.metal.h); draw a direction toward
//           it, pdf p (contract 3); if it gives no light (L_e = 0, a
//           firefly not yet awake, core/animation/glow.h), stop there, as
//           pbrt-v4's SimplePathIntegrator does (its test of ls->L): the
//           term is 0 whatever the shadow ray finds, so the ray is not
//           traced; else trace one shadow ray, to the light's surface, the
//           light itself ignored; if nothing blocks it,
//             L += beta x f(wo, wi) |cos(wi)| x L_e x T / (P x p),
//           T the transmittance of the medium the path is in over the
//           shadow ray's length: 1 in air; a light inside the same glass as
//           the surface, dimmed as the glass dims.
//           The cost per surface is the same for any number of lights
//           (principle 9).
//   Step 6  Sample the BSDF: wi, f and pdf (contract 2). If pdf is 0, stop.
//           beta *= f |cos(wi)| / pdf. The path counts emission at the next
//           surface only if this lobe was delta. If the lobe was a
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
//   Step 8  Continue: the ray leaves the surface along wi.
//
// What the camera ray starts from, and what is done with L, are the pass's:
// a point in the pixel drawn anew each frame, and the mean folded into the
// accumulated image (metal/film/accumulate.metal.h).
//
// The numbers a path draws are a function of the pixel, the frame's index
// and the dimension, the count of numbers the path drew before
// (metal/sampler/sampler.metal.h): independent between frames, so their mean
// converges, and any frame can be rendered again exactly, alone (principle 2).
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
// practice; 512 rays at the safety stop.
//
// The split, for ReSTIR DI (designed here, made with its implementation;
// the code below is unchanged until then). The loop above becomes two
// functions, declared at the end, that radiance() calls in turn, so the
// reference and ReSTIR share every line of light transport but the direct
// light at the first rough surface (logical-overview.md, principle 3; ES.3):
//
//   camera_segment()  steps 1 to 4 and, at surfaces with no lobe that aims
//                     at lights, 6 to 8, from the camera ray to the first
//                     surface that aims at lights (aims_at_lights), stopping
//                     there after step 4: its Shading, and the path's state,
//                     L, beta, eta_scale, the surface's count, and the
//                     numbers' counter, which the caller holds. Or the
//                     path's end before any such surface (it left the scene,
//                     reached a light, or roulette or the safety stop ended
//                     it among glass and mirrors), with L.
//   continue_path()   from that surface, step 5 there as `first` says, then
//                     steps 6 to 8 there and the loop on from the next
//                     surface, exactly as the loop runs now.
//
// Three callers: the path pass calls both, with `first` = aimed, so step 5
// runs at the surface as it does now, drawing the same numbers in the same
// order, and its image is unchanged to the bit; held by comparing the path
// pass's accumulated image before and after the split on every scene, and
// its frame time against its baseline, measured again
// (docs/research/2026-10-10-restir-di-layout.md, "Additive"). The surfaces
// pass calls camera_segment() and writes its result as the pixel's surface
// record and path state (contracts/surface_record.h); the shade calls
// continue_path() on what it reads back, with `first` = reused, so step 5 is
// skipped there and ReSTIR's direct light is the shade's to add
// (restir_di.metal.h, step 6). Its numbers continue the path's stream from
// where the segment left it: the three step 5 would have drawn are not
// drawn, which changes which numbers later steps draw, not their
// independence.
//
// Principle 7 holds through the split: at the first rough surface the
// direct light is counted once, by the naive step 5 or by ReSTIR's shade,
// and the next surface counts emission only after a delta lobe, in either.
//
// What the continuation reads of the surface, beyond its Shading's BSDF, wo
// and medium: the point and geometric normal (leaving it), the flags (which
// side it was arrived at from) and interior (the medium a transmission
// enters). The shade rebuilds those from the record and path state; the
// SurfaceInteraction's material, primitive, object_position and shading
// normal it leaves 0, unread by the continuation, and that is a contract of
// continue_path(): it reads no other field of `here.surface`, and adding a
// read of one adds it to contract 5's path state.


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

// The radiance arriving along the camera's ray, by the algorithm above;
// `numbers` the path's own.
inline float3 radiance(Scene scene, CameraRay camera, thread PathNumbers& numbers) {
    float3 L = float3(0.0f);
    float3 beta = float3(1.0f);
    float eta_scale = 1.0f;                        // step 6: what transmissions scaled beta by, undone
    bool counts_emission = true;                   // the camera's own ray
    uint medium = serenity::contracts::no_medium;  // air, at the camera
    metal::raytracing::ray along(camera.origin, camera.direction, 0.0f, unbounded);
    // Where the ray truly left from: the camera, then each surface's point.
    // The ray itself starts ray_offset off that surface, so its hit distance
    // is short of the stretch by that much; a medium's stretch is measured
    // from here.
    float3 from = camera.origin;

    for (uint surface = 0; surface < safety_stop; ++surface) {
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

        // Step 5.
        if (serenity::contracts::aims_at_lights(bsdf_lobes(here.bsdf)) && scene.selection().count > 0u) {
            L += beta * next_event(scene, here, numbers);
        }

        // Step 6.
        const serenity::contracts::BsdfSample next = bsdf_sample(here.bsdf, here.wo, next_numbers3(numbers));
        if (next.pdf <= 0.0f) {
            break;
        }
        const float3 wi = to_float3(next.direction);
        beta *= to_float3(next.value) * metal::abs(metal::dot(wi, to_float3(here.bsdf.normal))) / next.pdf;
        counts_emission = (next.lobe & serenity::contracts::lobe_delta) != 0u;
        if ((next.lobe & serenity::contracts::lobe_transmission) != 0u) {
            medium = (surface_at.flags & serenity::contracts::arrived_from_outside) != 0u
                         ? surface_at.interior
                         : serenity::contracts::no_medium;
            const float eta = bsdf_eta(here.bsdf, here.wo);
            eta_scale *= eta * eta;
        }

        // Step 7.
        if (surface >= roulette_from) {
            const float energy = metal::max(metal::max(beta.r, beta.g), beta.b) * eta_scale;
            const float q = metal::min(energy, max_survival);
            if (next_number(numbers) >= q) {
                break;
            }
            beta /= q;
        }

        // Step 8.
        from = point;
        along = leave(surface_at, wi);
    }
    return L;
}

// The split (above). Declared for review; radiance() is unchanged until the
// ReSTIR implementation makes it these two calls.

// Where the camera segment stopped, and the path's state there.
struct Segment {
    bool at_surface;   // false: the path ended before a surface that aims at lights; only L is meaningful
    Shading here;      // the surface, after step 4
    float3 L;          // the light found on the way
    float3 beta;       // throughput to the surface
    float eta_scale;   // step 6's
    uint surface;      // the surface's count along the path, from 0
};

// Step 5 at the segment's surface: aimed at a light, as the naive path
// tracer does, or left to ReSTIR's shade.
enum class FirstLight { aimed, reused };

inline Segment camera_segment(Scene scene, CameraRay camera, thread PathNumbers& numbers);

// The radiance the path gathers from `segment`'s surface on, L included.
// Precondition: segment.at_surface.
inline float3 continue_path(Scene scene, Segment segment, FirstLight first, thread PathNumbers& numbers);

}  // namespace path
}  // namespace shaders
}  // namespace serenity
