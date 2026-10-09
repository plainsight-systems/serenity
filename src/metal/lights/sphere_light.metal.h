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
// sample_light draws over the cone, pdf 1 / (its solid angle), and its
// radiance is the same everywhere on it and every way.
//
// Each takes the light as sphere_light() makes it from its data, its
// shape's transform and its glow, which the emitter reads from the frame's
// arrays (emitter.metal.h): the center is the transform's translation and
// the radius its scale (core/contracts/transform.h), wherever this frame
// placed it, and the radiance the light's peak times its glow this frame
// (core/animation/animate.h).

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/contracts/transform.h"
#include "core/lights/sphere_light.h"
#include "metal/device/layout.metal.h"
#include "metal/math/transform.metal.h"
#include "metal/math/warp.metal.h"

namespace serenity {
namespace shaders {

// A sphere light as the functions below read it: its data, and where its
// shape is and how big, from the frame's transforms (emitter.metal.h).
struct SphereLight {
    float3 center;
    float radius;
    float3 radiance;
    uint primitive;  // its shape, which a shadow ray toward it ignores
};

inline SphereLight sphere_light(serenity::lights::SphereLightData light, serenity::contracts::Transform placed,
                                float glow) {
    SphereLight s;
    s.center = transform_translation(placed);
    s.radius = transform_scale(placed);
    s.radiance = to_float3(light.radiance) * glow;
    s.primitive = light.shape;
    return s;
}

struct LightView {
    float3 direction;      // unit, from the point to the center
    float distance;        // to the center
    float sin2;            // sin^2 of the cone's half-angle: (r / d)^2, at most 1
    float cos_max;         // cos of the cone's half-angle
    float one_minus_cos;   // 1 - cos_max, computed without cancelling
    float solid_angle;     // of the cone: 2 pi (1 - cos_max)
};

inline LightView view_light(SphereLight light, float3 point) {
    const float3 to_center = light.center - point;
    LightView v;
    v.distance = metal::length(to_center);
    v.direction = to_center / v.distance;
    v.sin2 = metal::min(1.0f, (light.radius * light.radius) / (v.distance * v.distance));
    v.cos_max = metal::sqrt(1.0f - v.sin2);
    // 1 - cos a = sin^2 a / (1 + cos a): the direct difference cancels to 0
    // for a small, distant light (r / d = 1e-4 makes 1 - sin^2 round to 1),
    // and the light would vanish; this form keeps its precision.
    v.one_minus_cos = v.sin2 / (1.0f + v.cos_max);
    v.solid_angle = 2.0f * M_PI_F * v.one_minus_cos;
    return v;
}

// A direction within the light's cone, uniform in solid angle, from a point
// `u` in the unit square: cos t uniform on [cos_max, 1], the azimuth uniform.
inline float3 direction_to_light(LightView v, float2 u) {
    const float cos_t = 1.0f - u.x * v.one_minus_cos;
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
inline float distance_to_light(SphereLight light, LightView v, float3 direction) {
    const float along = v.distance * metal::dot(direction, v.direction);
    // The ray's distance from the center, squared: d^2 sin^2 of the angle
    // between them, from the cross product. Not d^2 - along^2, which for a
    // light far off subtracts two numbers near d^2 and loses the difference.
    const float3 across = metal::cross(direction, v.direction);
    const float off2 = v.distance * v.distance * metal::dot(across, across);
    return along - metal::sqrt(metal::max(0.0f, light.radius * light.radius - off2));
}

// Contract 3, for a sphere light. A point inside the light has no sample.

inline serenity::contracts::LightSample sphere_sample_light(SphereLight light, float3 point,
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
    s.radiance = to_packed(light.radiance);
    s.pdf = 1.0f / v.solid_angle;
    s.primitive = light.primitive;
    s.padding[0] = s.padding[1] = s.padding[2] = 0u;
    return s;
}

// The same radiance from every point of it, every way.
inline float3 sphere_emitted(SphereLight light) {
    return light.radiance;
}

inline float sphere_light_pdf(SphereLight light, float3 point, float3 direction) {
    const LightView v = view_light(light, point);
    // Inside the cone: on the light's side, and sin^2 of the angle to its
    // middle, the cross product's length squared, at most sin^2 a; this
    // holds its precision where cos against cos_max would cancel.
    const float3 across = metal::cross(direction, v.direction);
    if (v.distance <= light.radius || metal::dot(direction, v.direction) <= 0.0f ||
        metal::dot(across, across) > v.sin2) {
        return 0.0f;
    }
    return 1.0f / v.solid_angle;
}

}  // namespace shaders
}  // namespace serenity
