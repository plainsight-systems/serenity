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
// that surface again: leaving_point(), which leave() starts its ray at, so
// an estimator that must know where a ray will leave from before it has the
// ray (the path tracer's next event estimation, path.metal.h) asks the same
// function (ES.3). Two directions on the same side of the geometric normal
// (leave_alike()) leave from the same point, to the bit: the same
// expression on the same inputs, its one branch taken the same way.

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

// Whether `direction` leaves `surface` on the side its geometric normal
// points to (a direction in the tangent plane counts as that side).
inline bool leaves_outward(serenity::contracts::SurfaceInteraction surface, float3 direction) {
    return metal::dot(direction, to_float3(surface.geometric_normal)) >= 0.0f;
}

// Whether rays along `a` and `b` leave `surface` from the same point: both
// on the same side of its geometric normal.
inline bool leave_alike(serenity::contracts::SurfaceInteraction surface, float3 a, float3 b) {
    return leaves_outward(surface, a) == leaves_outward(surface, b);
}

// The point a ray along `direction` leaves `surface` from (above).
inline float3 leaving_point(serenity::contracts::SurfaceInteraction surface, float3 direction) {
    const float3 n = to_float3(surface.geometric_normal);
    return to_float3(surface.position) + ray_offset * (leaves_outward(surface, direction) ? n : -n);
}

// The ray leaving `surface` along unit `direction`, its hits up to `far`
// from where it starts.
inline metal::raytracing::ray leave(serenity::contracts::SurfaceInteraction surface, float3 direction,
                                    float far = unbounded) {
    return metal::raytracing::ray(leaving_point(surface, direction), direction, 0.0f, far);
}

}  // namespace shaders
}  // namespace serenity
