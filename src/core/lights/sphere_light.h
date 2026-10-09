#pragma once

// Axis: Light (sphere).
//
// A glowing sphere: the fireflies, light kind sphere (light.h). A sphere
// shape that wears an emissive material is one (materials/emissive.h); the
// scene reader lists each, with a light record for it, so a shader finds
// every light without searching the shapes. It is also a shape,
// so rays that reach it see its glow, and it casts shadows like any other.
//
// The light at a point p with normal n, from a sphere of radius r and
// radiance L whose center is at distance d, wholly above p's horizon, is the
// irradiance
//
//   E = pi L sin^2(a) cos(t),   sin(a) = r / d
//
// with t the angle between n and the direction to the center: exact for a
// uniform sphere, whose solid angle is a cap of half-angle a. How much of it
// reaches p is the fraction of the sphere visible from p
// (metal/lights/sphere_light.metal.h).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace lights {

struct SphereLightData {
    contracts::Float3 center;
    float radius;
    contracts::Float3 radiance;
    uint32_t primitive;  // the sphere's primitive, so a ray that reaches it is known to
};

static_assert(sizeof(SphereLightData) == 32, "SphereLightData must be the same 32 bytes on the host and in shaders");

}  // namespace lights
}  // namespace serenity
