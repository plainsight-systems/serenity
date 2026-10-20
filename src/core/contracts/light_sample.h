#pragma once

// Contract 4: the light sample. Owned by Light; read by Sample reuse and the
// Integrator.
//
// A light sample by identity: which light, and which point on it, in the
// light's own coordinates (logical-overview.md, principle 6). A path
// reservoir keeps one where its path reconnects to a light (contract 15,
// core/contracts/path_reservoir.h) from frame to frame and pixel to pixel,
// and every shift that reads it asks the light where that
// point is now, how bright, and how it faces the surface asking: the
// fireflies move between frames, so a position kept from last frame is
// wrong in this one, and an identity is right in every frame. Named
// LightPoint for what it holds: contract 3's LightSample
// (contracts/emitter.h) is the direction drawn toward a light from one
// point, which no other point can reuse.
//
// Stored as NVIDIA's RTXDI stores its light samples (lightData and uvData:
// an index with a valid bit, and a point in two 16-bit numbers), 8 bytes:
//
//   light  bit 31 set when the identity holds a sample (light_valid), and
//          then bits 0 to 30 the index of its light record
//          (core/lights/light.h), which says which kind and which of that
//          kind's lights. 0 for none: an empty reservoir's.
//   point  where on the light, as the light's kind places it in its own
//          coordinates, in two 16-bit numbers, x in the low half and y in
//          the high. For a sphere, the unit direction from its center in
//          its own space (contract 10: before its transform turns and
//          places it), which is the point on its unit sphere, mapped to
//          the square by the equal-area octahedral map (Clarberg 2008,
//          "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD") and
//          each of the square's coordinates, in [-1, 1], rounded to
//          round((c + 1) / 2 x 65535). The map and its inverse are shader
//          mathematics every family whose points are unit directions may
//          use (metal/math/warp.metal.h).
//
// Equal-area, not Cigolle et al.'s plain octahedral map (2014): every one
// of the 2^32 cells covers the same area of the unit sphere, 4 pi / 2^32,
// the measure a reconnection's density is in (metal/integrator/
// shift.metal.h), so the quantization below is the same size on every part
// of every light. It costs a square root more to decode.
//
// The quantization, stated (ES.46: a deliberate narrowing, its loss said).
// A point is drawn continuously and kept as its cell's representative,
// which every pass then evaluates, the candidates' pass too, before it
// weighs it: the point weighed, merged and shaded is one point everywhere,
// decoded where it is loaded and never stored decoded (GDSA.18: decoded in
// the consumer's load path). A cell is about 5 x 10^-5 radians across on
// the unit sphere, and a point's round trip, the angle from it to its
// cell's representative, at most unit_round_trip_error, 10^-4 (6.3 x 10^-5
// measured over the sphere, where the map's shear stretches a cell; below),
// so under 10^-4 of the light's radius in world space.
// What it does to the estimator is a small bias, which production accepts
// (RTXDI quantizes the same way): the sample's probability is its cell's,
// the draw's density integrated over the cell, while the estimator weighs
// it by the density and the target at the representative. For a density
// and an integrand that vary slowly across a cell that is the midpoint
// rule, accurate to the cell's size squared; where they do not, at the rim
// of the cap a surface sees, where the cosine at the light crosses 0
// inside a cell, the error is of the order of the share of the cap in rim
// cells, about the cell's size over the cap's angle. Not zero, and stated.
//
// The questions an emitter answers of an identity (metal/lights/
// emitter.metal.h), beside contract 3's, each kind answering for itself:
//
//   LightPoint light_point_at(light, index, reached)
//       the identity of `reached`, a point on light record `index`: a light
//       sample's point, or one a ray struck, encoded as above.
//   LightReceived evaluate_light_point(sample, point)
//       what the light at identity `sample` gives `point`, as this frame
//       places and lights it, at the decoded point: the direction and
//       distance to it, its radiance toward `point`, the cosine at the
//       light between its outward normal there and the direction back to
//       `point` (at most 0 where the point faces away, and then it gives
//       nothing), and its area scale, world area per unit of the light's
//       own area (a sphere's r^2: contract 10's transforms scale
//       uniformly). Evaluated against last frame's transforms and glows, it
//       is what last frame's surface saw.
//
// Densities over identities are per unit of the light's own area (the unit
// sphere's, for a sphere), so one sample means one thing at every surface
// and in every frame, whatever the light's scale then.
//
// Layout: two words, 8 bytes. Layout rules as for every shared contract
// (contracts/frame_constants.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// The equal-area octahedral map's round trip, at most: the angle, in
// radians, between a unit vector and its 32-bit code decoded (above;
// metal/math/warp.metal.h). Measured over the sphere by
// tests/gpu/encoding_test.cpp, which holds the map to it.
SERENITY_CONSTANT float unit_round_trip_error = 1.0e-4f;

// LightPoint::light's bits (above).
SERENITY_CONSTANT uint32_t light_valid = 0x80000000u;
SERENITY_CONSTANT uint32_t light_index_mask = 0x7fffffffu;

struct LightPoint {
  uint32_t light;  // light_valid | a light record's index; 0 for none
  uint32_t point;  // equal-area octahedral, x in bits 0-15, y in bits 16-31
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
static_assert(sizeof(LightPoint) == 8,
              "LightPoint must be the same 8 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightPoint>,
              "LightPoint is written to the GPU as bytes");
#endif

// Whether `sample` holds a light, and which record: the one reading of
// `light`'s bits, so no pass tests the bit its own way (ES.3).
SERENITY_SHARED_FUNCTION constexpr bool holds_light(LightPoint sample) {
  return (sample.light & light_valid) != 0u;
}
SERENITY_SHARED_FUNCTION constexpr uint32_t light_index(LightPoint sample) {
  return sample.light & light_index_mask;
}

// What evaluate_light_point() answers (above). Never stored: a shader's
// value, laid out here beside the identity it answers for, as contract 3's
// LightSample is.
struct LightReceived {
  Float3 direction;  // unit, from the point asking toward the light's point
  float distance;    // to it
  Float3 radiance;   // arriving along direction, if nothing is in the way
  float cos_light;   // at the light, its normal against -direction; <= 0 gives
                     // nothing
  float area_scale;  // world area per unit of the light's own area
  uint32_t primitive;  // the light's own shape, which a shadow ray ignores; or
                       // no_primitive
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): MSL has no std::array
  uint32_t padding[2];
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
    sizeof(LightReceived) == 48,
    "LightReceived must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightReceived>,
              "LightReceived is laid out as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
