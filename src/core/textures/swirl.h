#pragma once

// Axis: Texture (swirl).
//
// A marble's core: colored vanes twisting about its axis, as the cores of
// swirl and cat's-eye marbles do, two colors in bands that turn as they
// rise, their edges wavering. Laid on its shape, not on the world
// (texture.h): evaluated at the point in the shape's own coordinates, the
// unit sphere's for a sphere, so it turns, moves and scales with the core,
// and one swirl serves every core of its colors. A core is an opaque sphere
// inside a clear one (scenes/marbles.toml): the glass around it is the
// marble's, the swirl its color. Decided here, in the core, for every
// backend; the shader half is metal/textures/swirl.metal.h, and its tests
// check its properties on the GPU (noise.h says why).
//
// swirl(q), for q in the shape's own coordinates, by these steps, which the
// shader half carries by number:
//
//   Step 1  Where q is about the axis, y: its angle phi = atan2(q.z, q.x),
//           in (-pi, pi], and its height q.y.
//   Step 2  Its phase among the vanes: s = vanes (phi / (2 pi) + twist q.y)
//           + swirl_waver fbm(swirl_scale (cos phi, q.y, sin phi), 2, seed)
//           (noise.h): vanes bands around, a band's middle, where s is
//           constant, at phi = 2 pi (k / vanes - twist q.y), so each turns
//           twist times round per unit of height, whatever the vanes, its
//           edges wavering; the noise read on the unit cylinder at q's angle
//           and height.
//   Step 3  Its color: v = |2 fract(s) - 1|, 0 at a band of a's middle and 1
//           at one of b's, and the color a (1 - w) + b w, w =
//           smoothstep(0.5 - swirl_edge, 0.5 + swirl_edge, v): bands of each
//           color, their edges soft over 2 swirl_edge of a band.
//
// Every channel lies between a's and b's; it is the same at q and at any
// point on the axis's ray through q's angle and height (a function of phi
// and q.y, not of the distance from the axis), so a core's surface shows the
// bands as a solid core would.
//
// The scene's swirls are one array in this layout, bound at buffer 20 by
// every pass that reads the scene (passes/path/path.metal,
// passes/preview/preview.metal).
//
// Cost, per evaluation: an atan2, two noises and a few dozen flops, with no
// branch. Every tuning number is a named constant below (ES.45); its noise
// is keyed by (seed, lattice point), as noise.h's always is (GDSA.3).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#define SERENITY_CONSTANT constant constexpr
#else
#include <stdint.h>
#define SERENITY_CONSTANT inline constexpr
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace textures {

SERENITY_CONSTANT float swirl_waver = 0.15f;  // step 2: in bands
SERENITY_CONSTANT float swirl_scale = 2.0f;   // step 2: noise cells across the unit sphere
SERENITY_CONSTANT float swirl_edge = 0.08f;   // step 3: in bands

struct SwirlData {
    contracts::Float3 a;  // linear RGB in [0, 1]
    uint32_t vanes;       // bands of each color around the axis; 1 to 16
    contracts::Float3 b;  // linear RGB in [0, 1]
    float twist;          // turns per unit of height, in the shape's own coordinates; finite
    uint32_t seed;
    uint32_t padding[3];
};

static_assert(sizeof(SwirlData) == 48, "SwirlData must be the same 48 bytes on the host and in shaders");

}  // namespace textures
}  // namespace serenity

#undef SERENITY_CONSTANT
