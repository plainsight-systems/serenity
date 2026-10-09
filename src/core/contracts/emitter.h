#pragma once

// Contract 3: the emitter. Owned by Light; read by Light selection, the
// Integrator and, later, Sample reuse.
//
// How a light gives light to a point: a direction toward it drawn from
// numbers, the radiance arriving along that direction, how far away the
// light's surface is along it, and the density of the draw. Every light kind
// answers the same questions, so an estimator aims at a light without naming
// its kind (pbrt-v4's Light: SampleLi, PDF_Li, L).
//
// The data here is shared; the functions are each backend's, written to the
// semantics below, one file per kind in metal/lights/ and a dispatch on the
// light's kind (metal/lights/emitter.metal.h). Each takes a light record
// (core/lights/light.h), which Light selection chose without knowing its
// kind, and the scene's light arrays.
//
//   LightSample sample_light(light, point, u)
//       a direction from `point` toward the light, drawn from u, two numbers
//       in [0, 1), with the radiance arriving along it, the distance to the
//       light's surface along it, and its pdf per unit solid angle. pdf 0
//       means no sample (the point is inside the light).
//   float light_pdf(light, point, direction)
//       the density with which sample_light() draws `direction` from
//       `point`: what ReSTIR's resampling weights need of a candidate, and
//       what any estimator that weighs two strategies against each other
//       would. The naive path tracer weighs none (logical-overview.md,
//       principle 7). 0 for a direction that misses the light.
//   float3 emitted(light, point on it, direction)
//       the radiance the light sends from that point along that direction.
//   LightExtent extent(light, point)
//       the light as seen from `point`, whole: the direction to its middle,
//       the distance to it, the solid angle it fills, the sine of its
//       angular radius, and its mean radiance over that solid angle. What a
//       glossy surface's highlight needs: a lobe widened by the light's
//       size, over its solid angle.
//   float3 irradiance(light, point, normal)
//       the irradiance the light gives a surface at `point` with `normal`,
//       nothing in the way: the integral of radiance times the cosine over
//       the light. Exact for the sphere light, pi L sin^2 a cos t while it is
//       wholly above the surface's horizon (lights/sphere_light.h).
//
// The last two are what an estimator that is not Monte Carlo over the light
// needs: the deterministic preview (metal/passes/preview/preview.h), which
// visits every light and computes each one's light, rather than drawing a
// direction toward one. With them it reads every light through this
// contract and names no light kind.
//
// A shadow ray toward a sample is blocked by anything nearer than the
// light's surface along its direction, and not by the light itself: a light
// that is also a shape (a glowing sphere) names that shape, `primitive`,
// which the ray ignores, and the ray stops at `distance`. Ignoring it, not
// stopping short of it, is what makes this exact: near a sphere's rim a
// ray grazes it, and where it meets it is computed with too little
// precision for any fixed margin to fall reliably short.
//
// The sphere light (lights/sphere_light.h) draws uniformly over the cone it
// fills from the point, pdf = 1 / (2 pi (1 - cos a)), sin a = r / d, and its
// radiance is the same everywhere on it and every way.
//
// A light sample is given the light: the probability of choosing that light
// among all of them is Light selection's, and an estimator divides by both.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

// The shape a light is not, for primitive below.
#if defined(__METAL_VERSION__)
constant constexpr uint32_t no_primitive = 0xffffffffu;
#else
constexpr uint32_t no_primitive = 0xffffffffu;
#endif

struct LightSample {
    Float3 direction;    // unit, from the point toward the light
    float distance;      // along it, to the light's surface
    Float3 radiance;     // arriving along it, if nothing is in the way
    float pdf;           // per unit solid angle, given the light; 0 for no sample
    uint32_t primitive;  // the light's own shape, which a shadow ray ignores; or no_primitive
    uint32_t padding[3];
};

static_assert(sizeof(LightSample) == 48, "LightSample must be the same 48 bytes on the host and in shaders");

struct LightExtent {
    Float3 direction;     // unit, from the point toward the light's middle
    float distance;       // to its middle
    Float3 radiance;      // the mean over the solid angle it fills
    float solid_angle;    // the solid angle it fills from the point
    float sin_radius;     // the sine of its angular radius
    uint32_t primitive;   // its own shape, which a shadow ray ignores; or no_primitive
    uint32_t padding[2];
};

static_assert(sizeof(LightExtent) == 48, "LightExtent must be the same 48 bytes on the host and in shaders");

}  // namespace contracts
}  // namespace serenity
