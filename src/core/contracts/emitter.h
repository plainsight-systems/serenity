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
//
//       u = (0, 0) draws the light's middle as seen from the point: for a
//       sphere, the direction to its center. Every kind maps u so, and an
//       estimator that aims at a light with one fixed sample, as the
//       deterministic preview does, passes (0, 0) and relies on this alone,
//       never on a kind's mapping (I.1, I.5).
//   float light_pdf(light, point, direction)
//       the density with which sample_light() draws `direction` from
//       `point`: what ReSTIR's resampling weights need of a candidate, and
//       what any estimator that weighs two strategies against each other
//       would. The naive path tracer weighs none (logical-overview.md,
//       principle 7). 0 for a direction that misses the light.
//   float3 emitted(light, point on it, direction)
//       the radiance the light sends from that point along that direction.
//   bool light_at(primitive) -> light
//       whether a shape is a light, and which (core/lights/light.h,
//       shape_lights): how a path that reaches a surface learns it reached a
//       light, and asks emitted() its radiance, without naming a material.
//       What the light's surface scatters, if anything, is its BSDF's
//       (contract 2): a glowing sphere's scatters nothing.
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
// fills from the point, cos t = 1 - u.x (1 - cos a) from the cone's axis, so
// u.x = 0 is the axis, toward the center, pdf = 1 / (2 pi (1 - cos a)),
// sin a = r / d, and its
// radiance is the same everywhere on it and every way.
//
// A light sample is given the light: the probability of choosing that light
// among all of them is Light selection's, and an estimator divides by both.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

// The shape a light is not, for primitive below.
SERENITY_CONSTANT uint32_t no_primitive = 0xffffffffu;

struct LightSample {
    Float3 direction;    // unit, from the point toward the light
    float distance;      // along it, to the light's surface
    Float3 radiance;     // arriving along it, if nothing is in the way
    float pdf;           // per unit solid angle, given the light; 0 for no sample
    uint32_t primitive;  // the light's own shape, which a shadow ray ignores; or no_primitive
    uint32_t padding[3];
};

static_assert(sizeof(LightSample) == 48, "LightSample must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightSample>, "LightSample is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
