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
//   Step 1  Trace: the nearest surface along the ray, t away; beta *=
//           transmittance(medium, t) (contract 12), what the medium the ray
//           crossed kept of it: 1 in air.
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
//           it, pdf p (contract 3); trace one shadow ray, to the light's
//           surface, the light itself ignored; if nothing blocks it,
//             L += beta x f(wo, wi) |cos(wi)| x L_e / (P x p).
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
//           a marble's few pixels (GPU.4).
//   Step 7  Russian roulette, from the 4th surface: survive with
//           q = min(the largest channel of beta, 0.95), else stop;
//           beta /= q, so the mean is unchanged. This, not a depth limit,
//           ends paths: every path length keeps a chance, so the mean
//           converges to the full light transport (Veach 1997, 2.4).
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

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/surface_interaction.h"
#include "core/lights/gradient_sky.h"
#include "core/materials/material.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/light_selection/uniform_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/shapes/shapes.metal.h"
#include "metal/textures/textures.metal.h"

namespace serenity {
namespace shaders {
namespace path {

// How far a ray that leaves a surface starts off it, along the side's
// normal, so it does not hit that surface again: far below the scene's
// scale, far above float rounding there.
constant constexpr float offset = 1e-4f;
constant constexpr uint roulette_from = 3;  // the 4th surface, counting from 0
constant constexpr uint safety_stop = 256;
constant constexpr float max_survival = 0.95f;

struct Scene {
    metal::raytracing::primitive_acceleration_structure structure;
    Shapes shapes;
    Materials materials;
    Textures textures;
    UniformLight selection;
    Lights lights;
    serenity::lights::GradientSkyData sky;
};

// A point off the surface whose geometric normal is `n`, on the side
// `direction` leaves toward.
inline float3 leave(float3 point, float3 n, float3 direction) {
    return point + offset * (metal::dot(direction, n) >= 0.0f ? n : -n);
}

// The radiance arriving along the ray from `origin` along unit `direction`,
// the camera's, by the algorithm above; `numbers` the path's own.
inline float3 radiance(Scene scene, float3 origin, float3 direction, thread PathNumbers& numbers) {
    float3 L = float3(0.0f);
    float3 beta = float3(1.0f);
    bool counts_emission = true;  // the camera's own ray

    for (uint surface = 0; surface < safety_stop; ++surface) {
        // Step 1: Trace: the nearest surface along the ray.
        const Hit hit = trace(scene.structure, scene.shapes, origin, direction, 0.0f, INFINITY);

        // Step 2: Escape: the sky, the one way it is counted.
        if (!hit.found) {
            L += beta * gradient_sky(scene.sky, direction);
            break;
        }
        const float3 point = origin + hit.t * direction;
        const serenity::contracts::SurfaceInteraction surface_at =
            surface_interaction(scene.shapes, hit.primitive, point, direction);

        // Step 3: Emission, through the emitter (contract 3): counted on the
        // camera's ray and after a delta lobe; after any other bounce, step 5
        // at the surface before counted it. What the light's surface
        // scatters is its BSDF's: a glowing sphere's scatters nothing, so its
        // path ends at step 6.
        serenity::lights::LightRecord glowing;
        if (counts_emission && light_at(scene.lights, hit.primitive, glowing)) {
            L += beta * light_emitted(scene.lights, glowing, point, -direction);
        }

        // Step 4: Resolve the surface's BSDF (contract 2).
        const serenity::contracts::Bsdf bsdf = resolve_bsdf(scene.materials, scene.textures, surface_at);
        const float3 wo = -direction;
        const float3 n = to_float3(surface_at.geometric_normal);
        const float3 shading = to_float3(bsdf.normal);

        // Step 5: Next event estimation: one light, chosen uniformly, one
        // direction toward it, one shadow ray to its surface.
        if (serenity::contracts::aims_at_lights(bsdf_lobes(bsdf)) && scene.selection.count > 0u) {
            const SelectedLight chosen = select_light(scene.selection, next_number(numbers));
            const serenity::contracts::LightSample sample =
                sample_light(scene.lights, chosen.light, point, next_numbers2(numbers));
            if (sample.pdf > 0.0f) {
                const float3 wi = to_float3(sample.direction);
                const float3 f = bsdf_evaluate(bsdf, wo, wi);
                if (metal::any(f > 0.0f) && !occluded(scene.structure, scene.shapes, leave(point, n, wi), wi, 0.0f,
                                                      sample.distance, sample.primitive)) {
                    L += beta * f * metal::abs(metal::dot(wi, shading)) * to_float3(sample.radiance) /
                         (chosen.probability * sample.pdf);
                }
            }
        }

        // Step 6: Sample the BSDF for the next direction.
        const serenity::contracts::BsdfSample next = bsdf_sample(bsdf, wo, next_numbers3(numbers));
        if (next.pdf <= 0.0f) {
            break;
        }
        const float3 wi = to_float3(next.direction);
        beta *= to_float3(next.value) * metal::abs(metal::dot(wi, shading)) / next.pdf;
        counts_emission = (next.lobe & serenity::contracts::lobe_delta) != 0u;

        // Step 7: Russian roulette, from the 4th surface.
        if (surface >= roulette_from) {
            const float q = metal::min(metal::max(metal::max(beta.r, beta.g), beta.b), max_survival);
            if (next_number(numbers) >= q) {
                break;
            }
            beta /= q;
        }

        // Step 8: Continue along wi.
        origin = leave(point, n, wi);
        direction = wi;
    }
    return L;
}

}  // namespace path
}  // namespace shaders
}  // namespace serenity
