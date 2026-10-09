#pragma once

// Axis: Integrator (direct light, deterministic).
//
// The light arriving along a camera ray, counting what reaches each surface
// straight from the lights and the sky, and what metal and glass reflect and
// glass refracts: the estimator metal/passes/preview/preview.h describes,
// which states what it leaves out. Lights are chosen by the every-light
// selection (light_selection/every_light.metal.h); each sample's numbers
// come from the sampler (sampler/sampler.metal.h), by pixel and purpose.
//
// Materials are dispatched here by kind until the BSDF contract (contract
// 2) is written, with the path tracer: then each kind's evaluation and
// sampling come through it, and this switch goes.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/lights/gradient_sky.h"
#include "core/lights/light.h"
#include "core/materials/emissive.h"
#include "core/materials/material.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/light_selection/every_light.metal.h"
#include "metal/lights/emitter.metal.h"
#include "metal/lights/gradient_sky.metal.h"
#include "metal/materials/conductor.metal.h"
#include "metal/materials/dielectric.metal.h"
#include "metal/materials/rough.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "metal/shapes/shapes.metal.h"
#include "metal/textures/textures.metal.h"

namespace serenity {
namespace shaders {
namespace direct {

// How far a ray that leaves a surface starts off it, along the normal, so
// it does not hit that surface again: far below the scene's scale (a scene
// unit is about a meter), far above float rounding there.
constant constexpr float offset = 1e-4f;
constant constexpr uint max_glass = 8;

// Samples per pixel position, for each estimate.
constant constexpr uint light_samples = 4;  // per light: shadow rays across its cone
constant constexpr uint sky_samples = 2;    // rays toward the sky
constant constexpr uint gloss_samples = 4;  // rays a rough metal reflects

// What a sample is for: each purpose draws its own numbers (sampler.metal.h).
enum Purpose : uint {
    purpose_sky = 1,
    purpose_gloss = 2,
    purpose_light = 16,  // + the light's index
};

struct Scene {
    metal::raytracing::primitive_acceleration_structure structure;
    Shapes shapes;
    Textures textures;
    device const serenity::materials::MaterialRecord* materials;
    device const serenity::materials::RoughData* rough;
    device const serenity::materials::DielectricData* dielectrics;
    device const serenity::materials::ConductorData* conductors;
    device const serenity::materials::EmissiveData* emissives;
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
struct Surface {
    bool found;
    float3 point;
    float3 normal;  // outward
    serenity::materials::MaterialRecord material;
    uint primitive;
};

inline Surface reach(Scene scene, float3 origin, float3 direction) {
    const Hit hit = trace(scene.structure, scene.shapes, origin, direction, 0.0f, INFINITY);
    Surface s;
    s.found = hit.found;
    if (!hit.found) {
        return s;
    }
    s.point = origin + hit.t * direction;
    s.normal = shape_normal(scene.shapes, hit.primitive, s.point);
    s.material = scene.materials[shape_material(scene.shapes, hit.primitive)];
    s.primitive = hit.primitive;
    return s;
}

// The fraction of `light` visible from `from`, by `count` shadow rays toward
// directions the emitter draws over it (contract 3; for a sphere, uniformly
// over its cone), each reaching the light unless a shape lies before the
// light's surface. Glass is opaque to them: the light glass would focus is
// a caustic, which this estimator leaves out.
inline float visible_fraction(Scene scene, float3 point, float3 from, serenity::lights::LightRecord light, Pixel px,
                              uint light_index, uint count) {
    const float2 offset_2d = sample_offset(px.pixel, purpose_light + light_index);
    uint seen = 0;
    for (uint i = 0; i < count; ++i) {
        const serenity::contracts::LightSample sample =
            sample_light(scene.lights, light, point, sample_2d(offset_2d, px.position * count + i));
        if (sample.pdf <= 0.0f) {
            continue;
        }
        // Blocked by anything before the light, not by the light itself.
        seen += occluded(scene.structure, scene.shapes, from, to_float3(sample.direction), 0.0f, sample.distance,
                         sample.primitive)
                    ? 0u
                    : 1u;
    }
    return float(seen) / float(count);
}

inline float3 surface_color(Scene scene, Surface s) {
    return rough_color(scene.rough[s.material.index], scene.textures, s.point);
}

// A rough surface's light: from each selected light, its irradiance by the
// fraction of it visible; and from the sky, by cosine-distributed rays that
// nothing stops; times albedo / pi. What other surfaces reflect onto it is
// not counted.
inline float3 shade_rough(Scene scene, Surface s, float3 albedo, Pixel px) {
    const float3 from = s.point + offset * s.normal;
    float3 light_in = float3(0.0f);
    for (uint j = 0; j < selected_count(scene.selection); ++j) {
        const serenity::lights::LightRecord light = selected(scene.selection, j);
        const float3 irradiance = light_irradiance(scene.lights, light, s.point, s.normal);
        if (metal::all(irradiance == 0.0f)) {
            continue;
        }
        const float seen = visible_fraction(scene, s.point, from, light, px, j, light_samples);
        // Albedo / pi times the irradiance; the albedo is applied below.
        light_in += irradiance * M_1_PI_F * seen / selection_probability(scene.selection, j);
    }
    // Each cosine-distributed ray that leaves the scene brings the sky's
    // radiance; their mean times the albedo is the sky's contribution.
    const float2 sky_offset = sample_offset(px.pixel, purpose_sky);
    float3 sky_in = float3(0.0f);
    for (uint i = 0; i < sky_samples; ++i) {
        const float3 d = cosine_direction(s.normal, sample_2d(sky_offset, px.position * sky_samples + i));
        if (!occluded(scene.structure, scene.shapes, from, d, 0.0f, INFINITY, ~0u)) {
            sky_in += gradient_sky(scene.sky, d);
        }
    }
    return albedo * (light_in + sky_in / float(sky_samples));
}

// What a reflected ray shows, shaded more simply than what the camera sees
// (preview.h): a rough surface by each light whose center it sees and the
// unblocked sky above it; a glowing sphere by its glow when `see_glow`;
// metal and glass by the sky in the ray's direction.
inline float3 shade_reflected(Scene scene, float3 origin, float3 direction, bool see_glow) {
    const Surface s = reach(scene, origin, direction);
    if (!s.found) {
        return gradient_sky(scene.sky, direction);
    }
    switch (s.material.kind) {
    case serenity::materials::MaterialKind::emissive:
        return see_glow ? to_float3(scene.emissives[s.material.index].radiance) : float3(0.0f);
    case serenity::materials::MaterialKind::rough: {
        const float3 normal = metal::dot(s.normal, direction) > 0.0f ? -s.normal : s.normal;
        const float3 from = s.point + offset * normal;
        float3 light_in = gradient_sky(scene.sky, normal);
        for (uint j = 0; j < selected_count(scene.selection); ++j) {
            const serenity::lights::LightRecord light = selected(scene.selection, j);
            const float3 irradiance = light_irradiance(scene.lights, light, s.point, normal);
            const serenity::contracts::LightExtent extent = light_extent(scene.lights, light, s.point);
            // One shadow ray, toward the light's middle, ignoring the light.
            if (!metal::all(irradiance == 0.0f) &&
                !occluded(scene.structure, scene.shapes, from, to_float3(extent.direction), 0.0f, extent.distance,
                          extent.primitive)) {
                light_in += irradiance * M_1_PI_F / selection_probability(scene.selection, j);
            }
        }
        return surface_color(scene, s) * light_in;
    }
    case serenity::materials::MaterialKind::conductor:
    case serenity::materials::MaterialKind::dielectric:
        return gradient_sky(scene.sky, direction);
    }
    return float3(0.0f);
}

// A metal's light: each selected light's highlight, through GGX widened by
// the light's size (alpha' = alpha + sin(its angular radius) / 2, r / 2d
// for a sphere, so a near light's highlight is no smaller than the light)
// over the light's solid angle (contract 3, extent), by the fraction of it
// visible; and the rest of the scene, by rays the metal's microfacets
// reflect.
inline float3 shade_conductor(Scene scene, Surface s, float3 toward_eye, Pixel px) {
    const serenity::materials::ConductorData conductor = scene.conductors[s.material.index];
    const float alpha = metal::max(conductor.roughness * conductor.roughness, 1e-3f);
    const float3 from = s.point + offset * s.normal;

    float3 color = float3(0.0f);
    for (uint j = 0; j < selected_count(scene.selection); ++j) {
        const serenity::lights::LightRecord light = selected(scene.selection, j);
        const serenity::contracts::LightExtent extent = light_extent(scene.lights, light, s.point);
        const float widened = metal::min(1.0f, alpha + extent.sin_radius / 2.0f);
        const float3 reflectance =
            conductor_reflectance(conductor, widened, s.normal, toward_eye, to_float3(extent.direction));
        if (metal::all(reflectance == 0.0f)) {
            continue;
        }
        const float seen = visible_fraction(scene, s.point, from, light, px, j, light_samples);
        color += reflectance * to_float3(extent.radiance) * extent.solid_angle * seen /
                 selection_probability(scene.selection, j);
    }

    // The glow is not counted again here: the highlights above are it.
    const float2 gloss_offset = sample_offset(px.pixel, purpose_gloss);
    const float n_v = metal::dot(s.normal, toward_eye);
    float3 reflected = float3(0.0f);
    for (uint i = 0; i < gloss_samples; ++i) {
        const float3 h = sample_visible_normal(s.normal, toward_eye, alpha,
                                               sample_2d(gloss_offset, px.position * gloss_samples + i));
        const float3 l = metal::reflect(-toward_eye, h);
        const float n_l = metal::dot(s.normal, l);
        if (n_l <= 0.0f) {
            continue;
        }
        const float3 weight = schlick(to_float3(conductor.f0), metal::dot(toward_eye, h)) *
                              smith_g2(n_l, n_v, alpha) / smith_g1(n_v, alpha);
        reflected += weight * shade_reflected(scene, from, l, false);
    }
    return color + reflected / float(gloss_samples);
}

// The light arriving along one camera ray.
inline float3 radiance(Scene scene, float3 origin, float3 direction, Pixel px) {
    float3 color = float3(0.0f);
    float3 weight = float3(1.0f);
    for (uint glass = 0; glass < max_glass;) {
        const Surface s = reach(scene, origin, direction);
        if (!s.found) {
            return color + weight * gradient_sky(scene.sky, direction);
        }
        switch (s.material.kind) {
        case serenity::materials::MaterialKind::emissive:
            return color + weight * to_float3(scene.emissives[s.material.index].radiance);
        case serenity::materials::MaterialKind::rough: {
            Surface facing = s;
            facing.normal = metal::dot(s.normal, direction) > 0.0f ? -s.normal : s.normal;
            return color + weight * shade_rough(scene, facing, surface_color(scene, s), px);
        }
        case serenity::materials::MaterialKind::conductor:
            return color + weight * shade_conductor(scene, s, -direction, px);
        case serenity::materials::MaterialKind::dielectric: {
            const Boundary b = dielectric_boundary(scene.dielectrics[s.material.index], direction, s.normal);
            ++glass;
            if (b.total_internal) {
                origin = s.point + offset * b.facing;
                direction = b.reflected;
                break;
            }
            if (b.entering) {
                color +=
                    weight * b.reflectance * shade_reflected(scene, s.point + offset * b.facing, b.reflected, true);
            }
            weight *= 1.0f - b.reflectance;
            origin = s.point - offset * b.facing;
            direction = b.refracted;
            break;
        }
        }
    }
    return color;
}

}  // namespace direct
}  // namespace shaders
}  // namespace serenity
