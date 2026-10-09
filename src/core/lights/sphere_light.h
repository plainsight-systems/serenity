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
//
// Where it is and how big are its shape's: the translation and the scale of
// the transform that places the unit sphere (shapes/transform.h,
// shapes/sphere.h), read from the shapes' transforms, not copied here. A
// light samples its shape (change-axes.md), and a firefly that moves is
// moved once, as a shape (core/scene/animate.h), with its light following,
// and no second copy of its center to fall out of step. It is also the
// firefly's own coordinates, in which a reservoir keeps a sample on it
// (logical-overview.md, principle 6). What the light adds is its radiance.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace lights {

struct SphereLightData {
    contracts::Float3 radiance;
    uint32_t shape;  // its sphere: where it is and how big, and how a ray that reaches it knows it
};

static_assert(sizeof(SphereLightData) == 16, "SphereLightData must be the same 16 bytes on the host and in shaders");

}  // namespace lights
}  // namespace serenity
