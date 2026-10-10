#pragma once

// Axis: Integrator (shared by the estimators).
//
// What both estimators (path.metal.h, direct.metal.h) hold of a surface
// they shade, and how they leave it: one definition each (ES.3).
//
// Shading bundles what every question asked at a surface takes (I.23):
// where it is (contract 1), what it scatters (contract 2), the direction
// the light leaves toward, and the medium on that side (contract 12).
//
// A ray leaves a surface from a point `ray_offset` off it along its
// geometric normal, on the side the ray leaves toward, so it does not hit
// that surface again.

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/device/layout.metal.h"

namespace serenity {
namespace shaders {

struct Shading {
    serenity::contracts::SurfaceInteraction surface;
    serenity::contracts::Bsdf bsdf;
    float3 wo;    // unit, toward where the light goes
    uint medium;  // the medium on wo's side: an index into the media, or no_medium
};

// How far off the surface a leaving ray starts: far below the scene's scale
// (a scene unit is about a meter), far above float rounding there.
constant constexpr float ray_offset = 1e-4f;

// The ray leaving `surface` along unit `direction`, its hits up to `far`
// from where it starts.
inline metal::raytracing::ray leave(serenity::contracts::SurfaceInteraction surface, float3 direction,
                                    float far = unbounded) {
    const float3 n = to_float3(surface.geometric_normal);
    const float3 start = to_float3(surface.position) + ray_offset * (metal::dot(direction, n) >= 0.0f ? n : -n);
    return metal::raytracing::ray(start, direction, 0.0f, far);
}

}  // namespace shaders
}  // namespace serenity
