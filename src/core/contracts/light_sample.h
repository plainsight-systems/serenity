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
//   light  the index of its light record (core/lights/light.h): the record
//          says which kind and which of that kind's lights. no_light for a
//          reservoir that holds no sample.
//   point  where on the light, as the light's kind places it in its own
//          coordinates. For a sphere, the unit direction from its center in
//          its own space (contract 10: before its transform turns and
//          places it), which is the point on its unit sphere; encoded
//          octahedrally in two 16-bit numbers, x in the low half and y in
//          the high (Cigolle et al. 2014, "A Survey of Efficient
//          Representations for Independent Unit Vectors"; RTXDI keeps a
//          light's point in 32 bits the same way, its uvData):
//
//            p = v.xy / (|v.x| + |v.y| + |v.z|), folded where v.z < 0 to
//            (1 - |p.y|, 1 - |p.x|) with p's signs (0 counted positive);
//            each of p's coordinates, in [-1, 1], to round((c + 1) / 2 x
//            65535); decoded by the inverse and normalized.
//
// The encoding loses what is finer than its grid, 2 / 65535 on the
// octahedron's square, which is under 1e-4 radians on the sphere: under
// 0.15 micrometers on a firefly of 1.5 mm (ES.46: the narrowing deliberate,
// its loss stated). Every pass evaluates the decoded point, the candidates'
// pass too, before it weighs it, so the point weighed, merged and shaded is
// one point, the same everywhere; decoded where it is loaded, in the pass
// that reads it, never stored decoded (GDSA.18: compact codes decoded in
// the consumer's load path). What the grid does change is the density
// the point was drawn with: a candidate is drawn continuously and kept as
// its grid cell's point, so its density is the cell's mean, not its value
// at the point: a relative difference of about the cell's width, 1e-4
// radians, times the density's relative change per radian across the
// light, which is small but for points at the rim of the cap a surface
// sees. Stated, not hidden; not zero (metal/integrator/target.metal.h).
// The map itself is shader mathematics every family may use
// (metal/math/warp.metal.h).
//
// The questions an emitter answers of an identity (metal/lights/
// emitter.metal.h), beside contract 3's, each kind answering for itself:
//
//   LightPoint draw_light_point(light, point, u)
//       a point on light record `light`, drawn from `point` as sample_light()
//       draws a direction from it with the same u (contract 3), kept as an
//       identity: the point the drawn direction reaches on the light,
//       encoded as above. A point inside the light draws none (light
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
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"
#include "core/lights/light.h"

namespace serenity {
namespace contracts {

struct LightPoint {
    uint32_t light;  // a light record's index, or lights::no_light
    uint32_t point;  // octahedral, x in bits 0-15, y in bits 16-31
};

static_assert(sizeof(LightPoint) == 8, "LightPoint must be the same 8 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LightPoint>, "LightPoint is written to the GPU as bytes");
#endif

// What evaluate_light_point() answers (above). Never stored: a shader's value,
// laid out here beside the identity it answers for, as contract 3's
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
