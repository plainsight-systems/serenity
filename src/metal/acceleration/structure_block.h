#pragma once

// Axis: Acceleration (on the GPU: the structures a frame traces).
//
// What a pass binds to trace the scene, one shared layout written by the
// host (scene_acceleration.h) and read by the shaders (trace.metal.h),
// bound whole as `constant StructureBlock&` (passes/bindings.h). Apple's
// way of putting an acceleration structure in an argument buffer: the host
// writes its gpuResourceID, 64 bits, where the shader reads an
// acceleration structure (Metal Shading Language, 2.17). InstanceStructure
// and BoxStructure pick each side's type, type aliases rather than macros
// (ES.30), as core/scene/scene_block.h does.
//
//   scene           the instance structure over every shape's triangle
//                   mesh, instance i shape i, each carrying the mask bits
//                   of the questions it answers (everything_bit,
//                   occluder_bit: contract 19), traced by the intersector
//                   in hardware (docs/research/2026-10-23-apple-silicon-
//                   design.md, section 6, change 1).
//   seed_boxes      the occluders' bounding boxes, a primitive structure
//                   whose box i's per-primitive data is its shape's index:
//                   manifold next event estimation's seed traverses it
//                   once a segment, its intersection function visiting
//                   every box the segment enters (integrator/
//                   mnee.metal.h, seed_segment(); mnee_seed.metal). With no
//                   light, every shape's; with no surface, never traced.
//   occluder_count  how many shapes carry occluder_bit: none, and a shadow
//                   ray or the seed is not traced.
//
// The constant address space: 24 bytes every thread of every tracing pass
// reads before its traces, the same for all of them, the reuse that space
// is for (metal/scene/scene_block.metal.h, WWDC16 606). Fields largest
// first (CACHE.5), padded to 24 so the size is one statement on both
// sides. Written once a slot: at construction, and never again, since each
// slot's structure stays the same object however often it is rebuilt
// (scene_acceleration.h).

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
using InstanceStructure = metal::raytracing::instance_acceleration_structure;
using BoxStructure = metal::raytracing::primitive_acceleration_structure;
#else
using InstanceStructure = std::uint64_t;  // the structure's gpuResourceID
using BoxStructure = std::uint64_t;       // the structure's gpuResourceID
#endif

// The instances' mask bits (contract 19): every shape carries
// everything_bit; a shape that is not a light carries occluder_bit too.
// A namespace-scope constant is in the shading language's constant address
// space (core/contracts/shared_layout.h, SERENITY_CONSTANT).
//
// still_instance: the user id of an instance of a still structure, the
// still shapes' meshes placed in the world and concatenated, as Apple asks
// of instances that overlap (scene_acceleration.h); a hit there names its
// shape by its triangle (MeshTriangle::shape), a hit on any other instance
// by the instance's id, the shape's index.
#if defined(__METAL_VERSION__)
constant constexpr uint32_t everything_bit = 1u;
constant constexpr uint32_t occluder_bit = 2u;
constant constexpr uint32_t still_instance = 0xffffffffu;
#else
inline constexpr uint32_t everything_bit = 1u;
inline constexpr uint32_t occluder_bit = 2u;
inline constexpr uint32_t still_instance = 0xffffffffu;
#endif

struct StructureBlock {
  InstanceStructure scene;
  BoxStructure seed_boxes;
  uint32_t occluder_count;
  uint32_t padding;
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
    sizeof(StructureBlock) == 24,
    "StructureBlock must be the same 24 bytes on the host and in shaders");

// A triangle as a structure keeps it beside the hardware's own copy,
// Metal's per-primitive data (MTL4::AccelerationStructureTriangleGeometry
// Descriptor's primitiveDataBuffer), read from a hit's intersection result
// (trace.metal.h): its three vertices, in its mesh's space in a moving
// shape's mesh, in the world in a still structure, so a hit's point is the
// triangle's; and in a still structure its shape's index (still_instance,
// above), unread elsewhere. 40 bytes, packed floats and a word on both
// sides.
struct MeshTriangle {
  float v[3][3];
  uint32_t shape;
};

static_assert(sizeof(MeshTriangle) == 40,
              "MeshTriangle must be the same 40 bytes on the host and in "
              "shaders");

}  // namespace gpu
}  // namespace serenity
