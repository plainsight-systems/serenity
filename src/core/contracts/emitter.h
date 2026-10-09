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
//
// A shadow ray toward a sample stops short of `distance`: anything nearer
// than the light's surface along the direction blocks it, and the light's
// own surface does not.
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

struct LightSample {
    Float3 direction;  // unit, from the point toward the light
    float distance;    // along it, to the light's surface
    Float3 radiance;   // arriving along it, if nothing is in the way
    float pdf;         // per unit solid angle, given the light; 0 for no sample
};

static_assert(sizeof(LightSample) == 32, "LightSample must be the same 32 bytes on the host and in shaders");

}  // namespace contracts
}  // namespace serenity
