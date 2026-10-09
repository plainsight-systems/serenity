#pragma once

// Contract 1: the surface interaction. Owned by Shape; read by Material,
// Texture, the Integrator and, later, Sample reuse.
//
// Where a ray met a surface, as everything after the hit sees it: the
// point, the surface's normals, which side the ray arrived from, and what
// the surface is made of. A shape kind's intersection fills it
// (metal/shapes/), the material resolves its BSDF from it
// (contracts/bsdf.h), and ReSTIR keeps one per pixel to reuse samples
// between surfaces, which is why it is a shared layout and not only a
// shader struct.
//
// The normals point out of the shape, as the shape defines them, whichever
// side the ray arrived from; `flags` says which side that was. Keeping them
// unflipped lets glass tell entering from leaving, and lets a stored
// interaction be read without the ray that made it.
//
//   geometric_normal  the true surface's, for offsetting rays that leave it
//                     and for which side a direction is on;
//   shading_normal    the one materials shade with. Equal to the geometric
//                     normal for every shape kind so far; it differs once a
//                     kind interpolates normals or a material perturbs them.
//
// Surface coordinates (u, v) join this contract with the first texture that
// needs them; every texture so far is a function of position.
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

// Bits of SurfaceInteraction::flags.
#if defined(__METAL_VERSION__)
constant constexpr uint32_t arrived_from_outside = 1u;  // against the outward normal
#else
constexpr uint32_t arrived_from_outside = 1u;
#endif

struct SurfaceInteraction {
    Float3 position;
    uint32_t material;   // index into the material records (materials/material.h)
    Float3 geometric_normal;  // unit, outward
    uint32_t primitive;  // the shape, as primitive i is shape i (shapes/primitive.h)
    Float3 shading_normal;    // unit, outward
    uint32_t flags;      // arrived_from_outside, or 0
};

static_assert(sizeof(SurfaceInteraction) == 48,
              "SurfaceInteraction must be the same 48 bytes on the host and in shaders");

}  // namespace contracts
}  // namespace serenity
