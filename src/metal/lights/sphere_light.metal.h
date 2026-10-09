#pragma once

// Axis: Light (shader half of lights/sphere_light.h).
//
// How a glowing sphere looks from a point, before the surface there decides
// what it does with its light: the cone it fills, exactly, and directions
// within that cone for shadow rays.
//
// From a point at distance d from the center, a sphere of radius r fills a
// cone of half-angle a about the direction to its center, sin a = r / d:
// the cone of the rays tangent to it. Its solid angle is the cap
// 2 pi (1 - cos a). Shadow rays are drawn uniformly over that cone's solid
// angle, so the fraction of them that reach the light is the fraction of the
// light, as seen from the point, that is not hidden, out to its rim.
//
// Its answers to the emitter contract (contracts/emitter.h), at the end:
// sample_light draws over the cone, pdf 1 / (its solid angle); its radiance
// is the same everywhere and every way; its irradiance is pi L sin^2 a
// cos t, exact while the sphere is wholly above the surface's horizon. A
// sphere partly below the horizon is given the same formula, with cos t at
// least 0: an overestimate in that band, which only the preview's direct
// estimate reads (the path tracer aims rays instead).

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/lights/sphere_light.h"
#include "metal/device/layout.metal.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

struct LightView {
    float3 direction;   // unit, from the point to the center
    float distance;     // to the center
    float sin2;         // sin^2 of the cone's half-angle: (r / d)^2, at most 1
    float cos_max;      // cos of the cone's half-angle
    float solid_angle;  // of the cone: 2 pi (1 - cos_max)
};

inline LightView view_light(serenity::lights::SphereLightData light, float3 point) {
    const float3 to_center = to_float3(light.center) - point;
    LightView v;
    v.distance = metal::length(to_center);
    v.direction = to_center / v.distance;
    v.sin2 = metal::min(1.0f, (light.radius * light.radius) / (v.distance * v.distance));
    v.cos_max = metal::sqrt(1.0f - v.sin2);
    v.solid_angle = 2.0f * M_PI_F * (1.0f - v.cos_max);
    return v;
}

// A direction within the light's cone, uniform in solid angle, from a point
// `u` in the unit square: cos t uniform on [cos_max, 1], the azimuth uniform.
inline float3 direction_to_light(LightView v, float2 u) {
    const float cos_t = 1.0f - u.x * (1.0f - v.cos_max);
    const float sin_t = metal::sqrt(metal::max(0.0f, 1.0f - cos_t * cos_t));
    const float phi = 2.0f * M_PI_F * u.y;
    float3 t;
    float3 b;
    basis(v.direction, t, b);
    return sin_t * metal::cos(phi) * t + sin_t * metal::sin(phi) * b + cos_t * v.direction;
}

// How far along unit `direction`, within the light's cone, the ray reaches
// the light's surface from the point: the nearer root of the ray and the
// sphere. At the cone's rim the root is double, d cos a.
inline float distance_to_light(serenity::lights::SphereLightData light, LightView v, float3 direction) {
    const float along = v.distance * metal::dot(direction, v.direction);
    const float off2 = v.distance * v.distance - along * along;
    return along - metal::sqrt(metal::max(0.0f, light.radius * light.radius - off2));
}

// Contract 3, for a sphere light. A point inside the light has no sample.

inline serenity::contracts::LightSample sphere_sample_light(serenity::lights::SphereLightData light, float3 point,
                                                            float2 u) {
    serenity::contracts::LightSample s;
    const LightView v = view_light(light, point);
    if (v.distance <= light.radius) {
        s.direction = to_packed(float3(0.0f, 1.0f, 0.0f));
        s.distance = 0.0f;
        s.radiance = to_packed(float3(0.0f));
        s.pdf = 0.0f;
        s.primitive = light.primitive;
        s.padding[0] = s.padding[1] = s.padding[2] = 0u;
        return s;
    }
    const float3 d = direction_to_light(v, u);
    s.direction = to_packed(d);
    s.distance = distance_to_light(light, v, d);
    s.radiance = light.radiance;
    s.pdf = 1.0f / v.solid_angle;
    s.primitive = light.primitive;
    s.padding[0] = s.padding[1] = s.padding[2] = 0u;
    return s;
}

// The same radiance from every point of it, every way.
inline float3 sphere_emitted(serenity::lights::SphereLightData light) {
    return to_float3(light.radiance);
}

inline float sphere_light_pdf(serenity::lights::SphereLightData light, float3 point, float3 direction) {
    const LightView v = view_light(light, point);
    if (v.distance <= light.radius || metal::dot(direction, v.direction) < v.cos_max) {
        return 0.0f;
    }
    return 1.0f / v.solid_angle;
}

inline serenity::contracts::LightExtent sphere_extent(serenity::lights::SphereLightData light, float3 point) {
    const LightView v = view_light(light, point);
    serenity::contracts::LightExtent e;
    e.direction = to_packed(v.direction);
    e.distance = v.distance;
    e.radiance = light.radiance;
    e.solid_angle = v.solid_angle;
    e.sin_radius = metal::sqrt(v.sin2);
    e.primitive = light.primitive;
    e.padding[0] = e.padding[1] = 0u;
    return e;
}

inline float3 sphere_irradiance(serenity::lights::SphereLightData light, float3 point, float3 normal) {
    const LightView v = view_light(light, point);
    const float cos_t = metal::max(0.0f, metal::dot(normal, v.direction));
    return M_PI_F * to_float3(light.radiance) * v.sin2 * cos_t;
}

}  // namespace shaders
}  // namespace serenity
