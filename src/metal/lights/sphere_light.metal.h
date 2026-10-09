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

#include <metal_stdlib>

#include "core/lights/sphere_light.h"
#include "metal/device/layout.metal.h"
#include "metal/sampler/sampler.metal.h"

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

}  // namespace shaders
}  // namespace serenity
