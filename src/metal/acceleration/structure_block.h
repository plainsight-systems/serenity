#pragma once

// Axis: Acceleration (on the GPU: the structures a frame traces, by mask).
//
// What a pass binds to trace the scene: one structure for each ray mask
// (contract 19, core/contracts/ray_mask.h) and the table that names the
// shapes in the occluders' structure, one shared layout written by the host
// (scene_acceleration.h) and read by the shaders (trace.metal.h), bound
// whole as `constant StructureBlock&` where a single structure was bound
// before, at the same index (passes/bindings.h). Apple's way of putting an
// acceleration structure in an argument buffer: the host writes its
// gpuResourceID, 64 bits, where the shader reads a
// primitive_acceleration_structure (Metal Shading Language, 2.17); the
// table, as scene_block.h's arrays, a GPU address the shader reads as a
// pointer. Structure<T> and Array<T> pick each side's type, template
// aliases rather than macros (ES.30), as scene_block.h does.
//
//   everything        every shape, primitive i shape i (shapes/primitive.h),
//                     as before masks: the rays that carry everything trace
//                     exactly the structure they traced.
//   occluders         the surfaces alone, in shape order: its primitive i is
//                     shape occluder_shapes[i]. With no light in the scene,
//                     the everything structure itself (no second build) and
//                     the table the identity; with no surface, the table
//                     empty and occluder_count 0, the structure everything's,
//                     never traced (Metal builds no empty structure), and a
//                     ray carrying occluders finds nothing.
//   occluder_shapes   that table, occluder_count entries.
//
// The constant address space: 32 bytes every thread of every tracing pass
// reads before its queries, the same for all of them, the reuse that space
// is for (scene_block.h, WWDC16 606). Fields largest first, all 8 bytes but
// the count (CACHE.5), padded to 32 so the size is one statement on both
// sides. Written once a slot: at construction when nothing moves, and
// never again, since each slot's structures stay the same objects however
// often they are rebuilt (scene_acceleration.h).

#if defined(__METAL_VERSION__)
#include <metal_raytracing>
#include <metal_stdlib>
#else
#include <stdint.h>  // uint32_t in the global namespace, where the shading language has it

#include <cstdint>
#endif

namespace serenity {
namespace gpu {

#if defined(__METAL_VERSION__)
using Structure = metal::raytracing::primitive_acceleration_structure;
template <typename T>
using ShapeTable = constant T*;
#else
using Structure = std::uint64_t;  // the structure's gpuResourceID
template <typename T>
using ShapeTable = std::uint64_t;  // the table's GPU address
#endif

struct StructureBlock {
  Structure everything;
  Structure occluders;
  ShapeTable<uint32_t> occluder_shapes;
  uint32_t occluder_count;
  uint32_t padding;
};

static_assert(
    sizeof(StructureBlock) == 32,
    "StructureBlock must be the same 32 bytes on the host and in shaders");

}  // namespace gpu
}  // namespace serenity
