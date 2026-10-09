#pragma once

// Axis: Integrator (direct light, deterministic).
//
// The light arriving along a camera ray, counting what reaches each surface
// straight from the lights and the sky, and what glossy and delta surfaces
// reflect and refract: the estimator metal/passes/preview/preview.h
// describes, which states what it leaves out. Lights are chosen by the
// every-light selection (light_selection/every_light.metal.h); each sample's
// numbers come from the sampler (sampler/sampler.metal.h), by pixel and
// purpose.
//
// It reads everything through the contracts: surfaces through the surface
// interaction (contract 1) and the BSDF (contract 2, resolve, evaluate,
// sample, lobes), lights through the emitter (contract 3, sample_light,
// emitted, light_at). It names no material kind and no light kind; what it
// does at a surface it decides by the BSDF's lobes alone.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/emitter.h"
#include "core/contracts/surface_interaction.h"
#include "core/lights/gradient_sky.h"
#include "core/lights/light.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/light_selection/every_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/shapes/shapes.metal.h"
#include "metal/textures/textures.metal.h"

namespace serenity {
namespace shaders {
namespace direct {

// How far a ray that leaves a surface starts off it, along the geometric
// normal on its side, so it does not hit that surface again: far below the
// scene's scale (a scene unit is about a meter), far above float rounding.
constant constexpr float offset = 1e-4f;
constant constexpr uint max_delta = 8;  // delta surfaces a camera ray passes

// Samples per pixel position, for each estimate.
constant constexpr uint light_samples = 4;    // per light: shadow rays over it
constant constexpr uint diffuse_samples = 2;  // a diffuse lobe's rays toward the sky
constant constexpr uint glossy_samples = 4;   // a glossy lobe's reflected rays

// The far end of [0, 1): a delta BSDF sampled at u.x = 0 and here yields
// each of its lobes in turn (contract 2: u.x chooses among a kind's lobes).
constant constexpr float last_below_one = 0.99999994f;

// What a sample is for: each purpose draws its own numbers (sampler.metal.h).
enum Purpose : uint {
    purpose_bsdf = 1,
    purpose_lobe = 2,
    purpose_light = 16,  // + the light's index
};

struct Scene {
    metal::raytracing::primitive_acceleration_structure structure;
    Shapes shapes;
    Materials materials;
    Textures textures;
    EveryLight selection;
    Lights lights;
    serenity::lights::GradientSkyData sky;
};

// Where a sample's numbers come from: the pixel, and which of its positions
// this is, so each estimate's samples spread over all the pixel's positions.
struct Pixel {
    uint2 pixel;
    uint position;
};

// What a ray reached.
struct Reached {
    bool found;
    float3 point;
    serenity::contracts::SurfaceInteraction surface;
};

inline Reached reach(Scene scene, float3 origin, float3 direction) {
    const Hit hit = trace(scene.structure, scene.shapes, origin, direction, 0.0f, INFINITY);
    Reached r;
    r.found = hit.found;
    if (hit.found) {
        r.point = origin + hit.t * direction;
        r.surface = surface_interaction(scene.shapes, hit.primitive, r.point, direction);
    }
    return r;
}

// A point off the surface whose geometric normal is `n`, on the side
// `direction` leaves toward.
inline float3 leave(float3 point, float3 n, float3 direction) {
    return point + offset * (metal::dot(direction, n) >= 0.0f ? n : -n);
}

// The light a surface scatters toward wo straight from every light: per
// light, `count` directions the emitter draws over it (contract 3; for a
// sphere, uniformly over its cone), each f(wo, wi) |cos| L / pdf where no
// shape but the light lies before it (glass among them: the light glass
// would focus is a caustic, which this estimator leaves out), averaged, and
// divided by the light's selection probability.
inline float3 from_lights(Scene scene, Reached at, serenity::contracts::Bsdf bsdf, float3 wo, Pixel px,
                          uint count) {
    const float3 n = to_float3(at.surface.geometric_normal);
    const float3 shading = to_float3(bsdf.normal);
    float3 total = float3(0.0f);
    for (uint j = 0; j < selected_count(scene.selection); ++j) {
        const serenity::lights::LightRecord light = selected(scene.selection, j);
        const float2 offset_2d = sample_offset(px.pixel, purpose_light + j);
        float3 sum = float3(0.0f);
        for (uint i = 0; i < count; ++i) {
            const serenity::contracts::LightSample sample =
                sample_light(scene.lights, light, at.point, sample_2d(offset_2d, px.position * count + i));
            if (sample.pdf <= 0.0f) {
                continue;
            }
            const float3 wi = to_float3(sample.direction);
            const float3 f = bsdf_evaluate(bsdf, wo, wi);
            if (metal::all(f == 0.0f) || occluded(scene.structure, scene.shapes, leave(at.point, n, wi), wi, 0.0f,
                                                  sample.distance, sample.primitive)) {
                continue;
            }
            sum += f * metal::abs(metal::dot(wi, shading)) * to_float3(sample.radiance) / sample.pdf;
        }
        total += sum / float(count) / selection_probability(scene.selection, j);
    }
    return total;
}

// What a reflected ray shows, shaded more simply than what the camera sees
// (preview.h): a light's glow when `see_glow`; a surface with a diffuse lobe
// by each light toward its middle, one shadow ray each, and the sky above it
// unblocked, f(wo, n) pi sky(n); any other surface by the sky along the ray.
inline float3 shade_reflected(Scene scene, float3 origin, float3 direction, bool see_glow) {
    const Reached at = reach(scene, origin, direction);
    if (!at.found) {
        return gradient_sky(scene.sky, direction);
    }
    float3 color = float3(0.0f);
    serenity::lights::LightRecord glowing;
    if (light_at(scene.lights, at.surface.primitive, glowing) && see_glow) {
        color += light_emitted(scene.lights, glowing, at.point, -direction);
    }
    const serenity::contracts::Bsdf bsdf = resolve_bsdf(scene.materials, scene.textures, at.surface);
    const uint lobes = bsdf_lobes(bsdf);
    if (lobes == 0u) {
        return color;  // scatters nothing: a light's own surface
    }
    if ((lobes & serenity::contracts::lobe_diffuse) == 0u) {
        return color + gradient_sky(scene.sky, direction);
    }
    const float3 wo = -direction;
    const float3 n = to_float3(at.surface.geometric_normal);
    const float3 shading = facing(bsdf, wo);
    for (uint j = 0; j < selected_count(scene.selection); ++j) {
        const serenity::lights::LightRecord light = selected(scene.selection, j);
        // u = (0, 0): toward the light's middle, for a sphere.
        const serenity::contracts::LightSample sample = sample_light(scene.lights, light, at.point, float2(0.0f));
        if (sample.pdf <= 0.0f) {
            continue;
        }
        const float3 wi = to_float3(sample.direction);
        const float3 f = bsdf_evaluate(bsdf, wo, wi);
        if (!metal::all(f == 0.0f) && !occluded(scene.structure, scene.shapes, leave(at.point, n, wi), wi, 0.0f,
                                                sample.distance, sample.primitive)) {
            color += f * metal::abs(metal::dot(wi, shading)) * to_float3(sample.radiance) / sample.pdf /
                     selection_probability(scene.selection, j);
        }
    }
    return color + bsdf_evaluate(bsdf, wo, shading) * M_PI_F * gradient_sky(scene.sky, shading);
}

// The light a surface with a lobe that is not delta sends toward wo: from
// the lights directly, and by rays its BSDF samples (contract 2), each
// weighted value |cos| / pdf. A diffuse lobe's rays bring the sky where they
// escape and nothing where they meet a surface (the preview counts no light
// one rough surface reflects onto another); a glossy lobe's bring what they
// reach, shaded as a reflection, lights not counted again, from_lights
// having counted them.
inline float3 shade_scattering(Scene scene, Reached at, serenity::contracts::Bsdf bsdf, float3 wo, uint lobes,
                               Pixel px) {
    float3 color = from_lights(scene, at, bsdf, wo, px, light_samples);
    const uint count = (lobes & serenity::contracts::lobe_glossy) != 0u ? glossy_samples : diffuse_samples;
    const float2 direction_offset = sample_offset(px.pixel, purpose_bsdf);
    const float2 lobe_offset = sample_offset(px.pixel, purpose_lobe);
    const float3 n = to_float3(at.surface.geometric_normal);
    const float3 shading = to_float3(bsdf.normal);
    float3 bounced = float3(0.0f);
    for (uint i = 0; i < count; ++i) {
        const uint index = px.position * count + i;
        const float3 u = float3(sample_2d(lobe_offset, index).x, sample_2d(direction_offset, index));
        const serenity::contracts::BsdfSample sample = bsdf_sample(bsdf, wo, u);
        if (sample.pdf <= 0.0f) {
            continue;
        }
        const float3 wi = to_float3(sample.direction);
        const float3 weight = to_float3(sample.value) * metal::abs(metal::dot(wi, shading)) / sample.pdf;
        const float3 from = leave(at.point, n, wi);
        if ((sample.lobe & serenity::contracts::lobe_diffuse) != 0u) {
            if (!occluded(scene.structure, scene.shapes, from, wi, 0.0f, INFINITY, ~0u)) {
                bounced += weight * gradient_sky(scene.sky, wi);
            }
        } else {
            bounced += weight * shade_reflected(scene, from, wi, false);
        }
    }
    return color + bounced / float(count);
}

// The light arriving along one camera ray. Through surfaces whose lobes are
// all delta (glass), every lobe is taken, each with its probability: the
// BSDF sampled at both ends of u.x yields each lobe, and the product of a
// delta lobe's probability and its weight is value |cos| (contract 2). A
// reflection is shown where the ray arrives from outside, shaded as a
// reflection; the transmitted part continues, up to max_delta surfaces.
inline float3 radiance(Scene scene, float3 origin, float3 direction, Pixel px) {
    float3 color = float3(0.0f);
    float3 weight = float3(1.0f);
    for (uint delta = 0; delta < max_delta; ++delta) {
        const Reached at = reach(scene, origin, direction);
        if (!at.found) {
            return color + weight * gradient_sky(scene.sky, direction);
        }
        const float3 wo = -direction;
        serenity::lights::LightRecord glowing;
        if (light_at(scene.lights, at.surface.primitive, glowing)) {
            color += weight * light_emitted(scene.lights, glowing, at.point, wo);
        }
        const serenity::contracts::Bsdf bsdf = resolve_bsdf(scene.materials, scene.textures, at.surface);
        const uint lobes = bsdf_lobes(bsdf);
        if (lobes == 0u) {
            return color;  // scatters nothing: a light's own surface
        }
        if (serenity::contracts::aims_at_lights(lobes)) {
            return color + weight * shade_scattering(scene, at, bsdf, wo, lobes, px);
        }

        const float3 n = to_float3(at.surface.geometric_normal);
        const float3 shading = to_float3(bsdf.normal);
        const serenity::contracts::BsdfSample first = bsdf_sample(bsdf, wo, float3(0.0f, 0.5f, 0.5f));
        const serenity::contracts::BsdfSample last = bsdf_sample(bsdf, wo, float3(last_below_one, 0.5f, 0.5f));
        const bool two_lobes = first.pdf > 0.0f && last.pdf > 0.0f && first.lobe != last.lobe;
        if (two_lobes) {
            const bool first_reflects = (first.lobe & serenity::contracts::lobe_reflection) != 0u;
            const serenity::contracts::BsdfSample reflected = first_reflects ? first : last;
            const serenity::contracts::BsdfSample passed = first_reflects ? last : first;
            if ((at.surface.flags & serenity::contracts::arrived_from_outside) != 0u) {
                const float3 r = to_float3(reflected.direction);
                color += weight * to_float3(reflected.value) * metal::abs(metal::dot(r, shading)) *
                         shade_reflected(scene, leave(at.point, n, r), r, true);
            }
            const float3 t = to_float3(passed.direction);
            weight *= to_float3(passed.value) * metal::abs(metal::dot(t, shading));
            origin = leave(at.point, n, t);
            direction = t;
        } else {
            const serenity::contracts::BsdfSample only = first.pdf > 0.0f ? first : last;
            if (only.pdf <= 0.0f) {
                return color;
            }
            const float3 d = to_float3(only.direction);
            weight *= to_float3(only.value) * metal::abs(metal::dot(d, shading));
            origin = leave(at.point, n, d);
            direction = d;
        }
    }
    return color;
}

}  // namespace direct
}  // namespace shaders
}  // namespace serenity
