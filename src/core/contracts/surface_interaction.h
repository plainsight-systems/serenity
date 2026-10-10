#pragma once

// Contract 1: the surface interaction. Owned by Shape; read by Material,
// Texture, the Integrator and, later, Sample reuse.
//
// Where a ray met a surface, as everything after the hit sees it: the
// point, in the world and in the shape's own coordinates, the surface's
// normals, which side the ray arrived from, what the surface is made of,
// and what fills the shape behind it. A shape kind's intersection fills it
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
//   shading_normal    the shape's own shading normal: equal to the
//                     geometric normal for every shape kind so far, and
//                     different once a kind interpolates normals. Only the
//                     shape writes it. A material that perturbs the normal
//                     (a normal map) does so when it resolves its BSDF, into
//                     Bsdf::normal (contracts/bsdf.h), which it owns.
//
// The point in the shape's own coordinates, object_position, is where the
// hit is on the shape's geometry before its transform places it (contract
// 10): on the unit sphere for a sphere, at the box's own corners' scale for
// a box. The shape's intersection, which tests the ray in those coordinates
// already, fills it; a texture laid on its shape reads it (textures/
// texture.h), so a marble core's swirl turns and moves with the core. It is
// the first surface coordinate a texture needed; (u, v) join with the first
// that needs them.
//
// interior is the medium inside the shape (contract 12), from its record
// (shapes/primitive.h), or contracts::no_medium: what a path that passes
// through this surface into the shape enters.
//
// Layout: four groups of a packed triple and a word, 16 bytes each, so no
// field pads another (CACHE.5); 64 bytes. In the path kernel each surface's
// interaction is live in registers through its shading, so its size is
// register pressure, judged by the frame's time, not by occupancy (GPU.3,
// GPU.10); what the last 16 bytes cost is measured in
// docs/research/2026-10-09-medium-cost.md.
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

// Bits of SurfaceInteraction::flags.
SERENITY_CONSTANT uint32_t arrived_from_outside = 1u;  // against the outward normal

struct SurfaceInteraction {
    Float3 position;
    uint32_t material;   // index into the material records (materials/material.h)
    Float3 geometric_normal;  // unit, outward
    uint32_t primitive;  // the shape, as primitive i is shape i (shapes/primitive.h)
    Float3 shading_normal;    // unit, outward
    uint32_t flags;      // arrived_from_outside, or 0
    Float3 object_position;   // the point in the shape's own coordinates (contract 10)
    uint32_t interior;   // the medium inside the shape (contract 12), or no_medium
};

static_assert(sizeof(SurfaceInteraction) == 64,
              "SurfaceInteraction must be the same 64 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<SurfaceInteraction>, "SurfaceInteraction is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
