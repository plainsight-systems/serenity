#pragma once

// Axis: Light (shader half of lights/sphere_light.h).
//
// How much of a glowing sphere's light reaches a point, before the surface
// there decides what it does with it: the solid angle the sphere fills, the
// direction to it, and the points on it a shadow ray is aimed at.

#include <metal_stdlib>

#include "core/lights/sphere_light.h"
#include "metal/device/layout.metal.h"
#include "metal/sampler/sampler.metal.h"

namespace serenity {
namespace shaders {

struct LightView {
    float3 direction;     // unit, from the point to the center
    float distance;       // to the center
    float sin2;           // sin^2 of the cap's half-angle: (r / d)^2
    float solid_angle;    // of the cap, pi sin^2 for a small light
};

inline LightView view_light(serenity::lights::SphereLightData light, float3 point) {
    const float3 to_center = to_float3(light.center) - point;
    LightView v;
    v.distance = metal::length(to_center);
    v.direction = to_center / v.distance;
    v.sin2 = metal::min(1.0f, (light.radius * light.radius) / (v.distance * v.distance));
    v.solid_angle = M_PI_F * v.sin2;
    return v;
}

// A point on the light's disk as seen from the point: the disk through its
// center facing the point, which is the sphere's silhouette from there. `u`
// is a point in the unit square.
inline float3 point_on_light(serenity::lights::SphereLightData light, LightView v, float2 u) {
    float3 t;
    float3 b;
    basis(v.direction, t, b);
    const float2 d = light.radius * concentric_disk(u);
    return to_float3(light.center) + d.x * t + d.y * b;
}

}  // namespace shaders
}  // namespace serenity
