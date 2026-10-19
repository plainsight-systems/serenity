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
//
// Its answers to contract 4, the light sample by identity
// (contracts/light_sample.h), declared at the end beside contract 3's and
// leaving them as they are, so the path tracer draws the numbers it drew
// and renders the image it rendered:
//
//   - a point found on the sphere kept as its identity: that point in the
//     sphere's own space, the unit direction R^T (y - c) / r from its
//     center (metal/math/transform.metal.h, transform_to_object), encoded
//     by the equal-area octahedral map (metal/math/warp.metal.h; contracts/
//     light_sample.h states what it loses). Its own space and not the
//     world's: a light that turns keeps its point, as one that moves does.
//     The scene's transforms do not turn yet (contracts/transform.h); the
//     identity is correct when they do.
//   - an identity evaluated at a point: the decoded unit direction placed by
//     this frame's transform, y = c + s R v; the direction and distance from
//     the point to y; the radiance (the same every way, its glow this
//     frame's); the cosine at y between the sphere's outward normal R v and
//     the direction back to the point, at most 0 for the half of the sphere
//     the point cannot see; and the area scale s^2, a sphere's world area
//     per unit of its own.
//
// What the cone draws, it draws only on the cap the point can see; an
// identity evaluated from another point may be on that point's far side,
// and gives it nothing (cos <= 0): the target's support, not a failure.

#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/contracts/light_sample.h"
#include "core/contracts/transform.h"
#include "core/lights/sphere_light.h"
#include "metal/device/layout.metal.h"
#include "metal/math/srgb.metal.h"
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

// Its power (emitter.metal.h, light_power): 4 pi r^2 x pi x luminance,
// RTXDI's SphereLight::GetPower without its shaping.
inline float sphere_light_power(SphereLight light) {
  constexpr float pi = 3.14159265358979f;
  return 4.0f * pi * light.radius * light.radius * pi *
         luminance(light.radiance);
}

inline SphereLight sphere_light(serenity::lights::SphereLightData light,
                                serenity::contracts::Transform placed,
                                float glow) {
  return SphereLight{transform_translation(placed), transform_scale(placed),
                     to_float3(light.radiance) * glow, light.shape};
}

struct LightView {
  float3 direction;     // unit, from the point to the center
  float distance;       // to the center
  float sin2;           // sin^2 of the cone's half-angle: (r / d)^2, at most 1
  float cos_max;        // cos of the cone's half-angle
  float one_minus_cos;  // 1 - cos_max, computed without cancelling
  float solid_angle;    // of the cone: 2 pi (1 - cos_max)
};

inline LightView view_light(SphereLight light, float3 point) {
  const float3 to_center = light.center - point;
  const float distance = metal::length(to_center);
  const float sin2 =
      metal::min(1.0f, (light.radius * light.radius) / (distance * distance));
  const float cos_max = metal::sqrt(1.0f - sin2);
  // 1 - cos a = sin^2 a / (1 + cos a): the direct difference cancels to 0
  // for a small, distant light (r / d = 1e-4 makes 1 - sin^2 round to 1),
  // and the light would vanish; this form keeps its precision.
  const float one_minus_cos = sin2 / (1.0f + cos_max);
  return LightView{
      to_center / distance,         distance, sin2, cos_max, one_minus_cos,
      2.0f * M_PI_F * one_minus_cos};
}

// A direction within the light's cone, uniform in solid angle, from a point
// `u` in the unit square: cos t uniform on [cos_max, 1], the azimuth uniform.
inline float3 direction_to_light(LightView v, float2 u) {
  const float cos_t = 1.0f - u.x * v.one_minus_cos;
  const float sin_t = metal::sqrt(metal::max(0.0f, 1.0f - cos_t * cos_t));
  const float phi = 2.0f * M_PI_F * u.y;
  const Tangents f = tangents(v.direction);
  return sin_t * metal::cos(phi) * f.t + sin_t * metal::sin(phi) * f.b +
         cos_t * v.direction;
}

// How far along unit `direction`, within the light's cone, the ray reaches
// the light's surface from the point: the nearer root of the ray and the
// sphere. At the cone's rim the root is double, d cos a.
inline float distance_to_light(SphereLight light, LightView v,
                               float3 direction) {
  const float along = v.distance * metal::dot(direction, v.direction);
  // The ray's distance from the center, squared: d^2 sin^2 of the angle
  // between them, from the cross product. Not d^2 - along^2, which for a
  // light far off subtracts two numbers near d^2 and loses the difference.
  const float3 across = metal::cross(direction, v.direction);
  const float off2 = v.distance * v.distance * metal::dot(across, across);
  return along -
         metal::sqrt(metal::max(0.0f, light.radius * light.radius - off2));
}

// Contract 3, for a sphere light. A point inside the light has no sample:
// LightSample{}, its pdf 0.

inline serenity::contracts::LightSample sphere_sample_light(SphereLight light,
                                                            float3 point,
                                                            float2 u) {
  const LightView v = view_light(light, point);
  if (v.distance <= light.radius) {
    return serenity::contracts::LightSample{};
  }
  const float3 d = direction_to_light(v, u);
  return serenity::contracts::LightSample{to_packed(d),
                                          distance_to_light(light, v, d),
                                          to_packed(light.radiance),
                                          1.0f / v.solid_angle,
                                          light.primitive,
                                          {0u, 0u, 0u}};
}

// The same radiance from every point of it, every way.
inline float3 sphere_emitted(SphereLight light) { return light.radiance; }

inline float sphere_light_pdf(SphereLight light, float3 point,
                              float3 direction) {
  const LightView v = view_light(light, point);
  // Inside the cone: on the light's side, and sin^2 of the angle to its
  // middle, the cross product's length squared, at most sin^2 a; this
  // holds its precision where cos against cos_max would cancel.
  const float3 across = metal::cross(direction, v.direction);
  if (v.distance <= light.radius ||
      metal::dot(direction, v.direction) <= 0.0f ||
      metal::dot(across, across) > v.sin2) {
    return 0.0f;
  }
  return 1.0f / v.solid_angle;
}

// Contract 4, for a sphere light placed by `placed` (the transform
// sphere_light() read its center and radius from). Pure (F.8).

// The point `reached`, on the sphere placed by `placed` up to rounding,
// encoded as contract 4's `point`: the unit direction from the center in
// the sphere's own space.
inline uint sphere_point_at(serenity::contracts::Transform placed,
                            float3 reached) {
  // On the unit sphere in its own space, up to the root's rounding.
  return encode_unit(metal::normalize(transform_to_object(placed, reached)));
}

// The density with which sphere_sample_light() draws the direction from
// `point` toward a point of the sphere that `received` describes, the
// point's own: 1 over the cone's solid angle wherever the point faces
// `point` (cos_light > 0), 0 behind and from inside. sphere_light_pdf()
// asks it of a direction, whose test against the cone's rim rounding can
// fail for a point on the rim itself; a point on the sphere is in the cone
// by construction, so this asks nothing of the direction (a shift that
// reconnects to a light point may land on the rim as anywhere:
// metal/integrator/shift.metal.h).
inline float sphere_point_pdf(SphereLight light, float3 point,
                              serenity::contracts::LightReceived received) {
  const LightView v = view_light(light, point);
  return v.distance > light.radius && received.cos_light > 0.0f
             ? 1.0f / v.solid_angle
             : 0.0f;
}

// What the sphere's point in `sample` gives `point`, as this light is
// placed and lit, at the decoded point. The identity, not its bare word,
// so it cannot be passed in another word's place (I.24).
inline serenity::contracts::LightReceived sphere_evaluate_point(
    SphereLight light, serenity::contracts::Transform placed,
    serenity::contracts::LightPoint sample, float3 point) {
  // s R v: the outward normal at y scaled by the radius, and y - c.
  const float3 outward = transform_direction(placed, decode_unit(sample.point));
  const float3 to_light = light.center + outward - point;
  const float distance = metal::length(to_light);
  const float3 direction = to_light / distance;
  return serenity::contracts::LightReceived{
      to_packed(direction),
      distance,
      to_packed(light.radiance),
      -metal::dot(metal::normalize(outward), direction),
      light.radius * light.radius,
      light.primitive,
      {0u, 0u}};
}

}  // namespace shaders
}  // namespace serenity
