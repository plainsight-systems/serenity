#pragma once

// Axis: Integrator (shared by the estimators).
//
// What the estimators (path.metal.h, ReSTIR PT's tree and shifts) hold of a
// surface they shade, and how they leave it: one definition each (ES.3).
//
// Shading bundles what every question asked at a surface takes (I.23):
// where it is (contract 1), what it scatters (contract 2), the direction
// the light leaves toward, and the medium on that side (contract 12).
//
// A ray leaves a surface from a point just off it along its geometric
// normal, on the side the ray leaves toward, so it does not hit that surface
// again: leaving_point(), which leave() starts its ray at, so an estimator
// that must know where a ray will leave from before it has the ray (the path
// tracer's next event estimation, path.metal.h) asks the same function
// (ES.3). Two directions on the same side of the geometric normal
// (leave_alike()) leave from the same point, to the bit: the same expression
// on the same inputs, its one branch taken the same way.
//
// How far off: by the float rounding of the point itself, Wachter and
// Binder's offset_ray() (Ray Tracing Gems, 2019, chapter 6). Each coordinate
// moves int_scale x its normal's component units in the last place (ulps),
// on the float's bits, so the step grows with the coordinate as its
// rounding does: 256 ulps, 15 um at the marbles' table, 0.75 m up, 2 mm at
// 100 m. Within origin_band of 0, where ulps shrink toward nothing and the
// error a coordinate carries is the scene's, not its own, by float_scale x
// the normal's component instead, a fixed 15 um. Every ray's hits count
// from 0 on: no minimum distance (below).
//
// What it assumes, and what holds it (tests/gpu/leave_test.cpp, both sides,
// head on to grazing, at the scenes' scales and past them):
//
//   - the point is on the surface to a few ulps of its own coordinates,
//     however far the ray that found it came. Falcor and RTXDI ask it of a
//     triangle's point interpolated from its vertices; ours are put back on
//     their shape by surface_interaction() (shapes/shapes.metal.h, contract
//     1), since a ray's origin + t d is off it by t's rounding, which grows
//     with t, and from far off is more than the step;
//   - the shape is larger than the step where it is: a sphere 2 mm across
//     100 m out is as wide as its step, and a ray leaving it inward starts
//     at its far side;
//   - a ray tested against the surface it leaves rounds within the step:
//     a box's test at its own coordinates does; a sphere's rounds with its
//     radius, so where the step is the fixed one, near the origin, a sphere
//     tens of meters across holds and one hundreds across does not.
//
// The chapter's constants, not the three times larger ones Falcor and
// RTXDI's path tracer use (1/16, 3/65536, 768): the step is also the light's
// error, every estimator seeing a light from the leaving point
// (path.metal.h, "One origin"), and the chapter's hold every case of the
// test. Alternatives considered (docs/research/2026-10-11-ray-offset.md):
// pbrt-v4's error bounds carried with each point (tighter still, at a
// float3 more in contract 1, which the path loop holds in registers); a
// fixed step (the 0.1 mm this replaces, seven times the step on the
// marbles' table and below one ulp 10 km out); tracing from the point
// itself with a minimum distance (RTXDI's visibility rays, 1 mm), which
// drops true hits nearer than it, a firefly resting on the table among
// them, and does not bound a grazing ray's self-hit, whose t's rounding
// grows as 1 / cos (the chapter's argument). RTXDI's path tracer adds a
// minimum distance of 10 um to Falcor's offset; with points put back on
// their shapes, the test needs none.
//
// Bit arithmetic on unsigned words (ES.101), so a coordinate within
// origin_band, whose integer step is computed and then not chosen, cannot
// overflow (ES.103); the float's bits by as_type, MSL's named cast (the
// repo's ES.49 deviation, AGENTS.md). Pure (F.8). Per ray, a few integer
// adds, compares and selects, no branch (GEN.2; GPU.4: no lane diverges on
// a coordinate's band). Its cost a frame is measured in the note; a float
// form of the same step (the exponent's bits times 2^-23) was no cheaper.

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
  float3 wo;  // unit, toward where the light goes
  // the medium on wo's side: an index into the media, or no_medium
  uint medium;
};

// offset_ray()'s constants (above; ES.45): the band about 0 given a fixed
// step, that step per unit of normal, and the step elsewhere in ulps per
// unit of normal.
constant constexpr float origin_band = 1.0f / 32.0f;
constant constexpr float float_scale = 1.0f / 65536.0f;
constant constexpr float int_scale = 256.0f;

// `p` moved off its surface along unit `n` by its own rounding (above).
inline float3 offset_ray(float3 p, float3 n) {
  const int3 steps = int3(int_scale * n);  // |steps| <= 256, toward zero
  // Adding to a float's bits moves it away from 0; so toward +n a negative
  // coordinate's bits take the step away.
  const uint3 bits = as_type<uint3>(p) +
                     as_type<uint3>(metal::select(steps, -steps, p < 0.0f));
  return metal::select(as_type<float3>(bits), p + float_scale * n,
                       metal::abs(p) < origin_band);
}

// Whether `direction` leaves `surface` on the side its geometric normal
// points to (a direction in the tangent plane counts as that side).
inline bool leaves_outward(serenity::contracts::SurfaceInteraction surface,
                           float3 direction) {
  return metal::dot(direction, to_float3(surface.geometric_normal)) >= 0.0f;
}

// Whether rays along `a` and `b` leave `surface` from the same point: both
// on the same side of its geometric normal.
inline bool leave_alike(serenity::contracts::SurfaceInteraction surface,
                        float3 a, float3 b) {
  return leaves_outward(surface, a) == leaves_outward(surface, b);
}

// The point a ray along `direction` leaves `surface` from (above).
inline float3 leaving_point(serenity::contracts::SurfaceInteraction surface,
                            float3 direction) {
  const float3 n = to_float3(surface.geometric_normal);
  return offset_ray(to_float3(surface.position),
                    leaves_outward(surface, direction) ? n : -n);
}

// The ray leaving `surface` along unit `direction`, its hits up to `far`
// from where it starts.
inline metal::raytracing::ray leave(
    serenity::contracts::SurfaceInteraction surface, float3 direction,
    float far = unbounded) {
  return metal::raytracing::ray(leaving_point(surface, direction), direction,
                                0.0f, far);
}

}  // namespace shaders
}  // namespace serenity
