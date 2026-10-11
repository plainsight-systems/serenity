#pragma once

// Contract 4: the light sample. Owned by Light; read by Sample reuse and the
// Integrator.
//
// A light sample by identity: which light, and which point on it, in the
// light's own coordinates (logical-overview.md, principle 6). A reservoir
// keeps one (contract 14, core/contracts/reservoir.h) from frame to frame and
// pixel to pixel, and every pass that reads it asks the light where that
// point is now, how bright, and how it faces the surface asking: the
// fireflies move between frames, so a position kept from last frame is
// wrong in this one, and an identity is right in every frame. Named
// LightPoint for what it holds: contract 3's LightSample
// (contracts/emitter.h) is the direction drawn toward a light from one
// point, which no other point can reuse.
//
//   point  where on the light, as the light's kind places it in its own
//          coordinates, at the full float precision the draw produced. For
//          a sphere, the unit direction from its center in its own space
//          (contract 10: before its transform turns and places it), which is
//          the point on its unit sphere: three floats.
//   light  the index of its light record (core/lights/light.h): the record
//          says which kind and which of that kind's lights. no_light for a
//          reservoir that holds no sample.
//
// Exactly, not quantized. A point snapped to a grid would make the sample
// the grid cell's representative, whose probability is the draw's density
// integrated over the cell: an estimator that weighs it by the density at
// the point, and shades the point, sums a quadrature over the cells, which
// no number of frames turns into the integral the reference converges to
// (principle 3). Kept as drawn, the point every pass evaluates is the
// sample the candidates weighed, and the estimator integrates the same
// continuous measure as the reference. What separates the drawn point from
// the stored one is float rounding in carrying it into the light's space,
// the precision every evaluation in the renderer has, the reference's too.
//
// The questions an emitter answers of an identity (metal/lights/
// emitter.metal.h), beside contract 3's, each kind answering for itself:
//
//   LightPoint draw_light_point(light, point, u)
//       a point on light record `light`, drawn from `point` as sample_light()
//       draws a direction from it with the same u (contract 3), kept as an
//       identity: the point the drawn direction reaches on the light, in
//       the light's own space. A point inside the light draws none (light
//       no_light). sample_light() itself is unchanged and draws the same
//       numbers it did.
//   LightReceived evaluate_light_point(sample, point)
//       what the light at identity `sample` gives `point`, as this frame
//       places and lights it: the direction and distance to it, its
//       radiance toward `point`, the cosine at the light between its
//       outward normal there and the direction back to `point` (at most 0
//       where the point faces away, and then it gives nothing), and its
//       area scale, world area per unit of the light's own area (a
//       sphere's r^2: contract 10's transforms scale uniformly). Evaluated
//       against last frame's transforms and glows, it is what last frame's
//       surface saw (metal/reuse/temporal.metal.h).
//
// Densities over identities are per unit of the light's own area (the unit
// sphere's, for a sphere), so one sample means one thing at every surface
// and in every frame, whatever the light's scale then
// (metal/integrator/target.metal.h).
//
// Layout: a packed triple and a word, 16 bytes (CACHE.5); contract 14 holds
// one at its start. Layout rules as for every shared contract
// (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"
#include "core/lights/light.h"

namespace serenity {
namespace contracts {

struct LightPoint {
    Float3 point;    // in the light's own space; for a sphere, unit length
    uint32_t light;  // a light record's index, or lights::no_light
};

static_assert(sizeof(LightPoint) == 16, "LightPoint must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightPoint>, "LightPoint is written to the GPU as bytes");
#endif

// What evaluate_light_point() answers (above). Never stored: a shader's
// value, laid out here beside the identity it answers for, as contract 3's
// LightSample is.
struct LightReceived {
    Float3 direction;    // unit, from the point asking toward the light's point
    float distance;      // to it
    Float3 radiance;     // arriving along direction, if nothing is in the way
    float cos_light;     // at the light, its normal against -direction; <= 0 gives nothing
    float area_scale;    // world area per unit of the light's own area
    uint32_t primitive;  // the light's own shape, which a shadow ray ignores; or no_primitive
    uint32_t padding[2];
};

static_assert(sizeof(LightReceived) == 48, "LightReceived must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightReceived>, "LightReceived is laid out as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
