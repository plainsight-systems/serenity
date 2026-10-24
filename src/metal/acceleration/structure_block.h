#pragma once

// Axis: Acceleration (on the GPU: the structures a frame traces).
//
// What a pass binds to trace the scene, one shared layout written by the
// host (scene_acceleration.h) and read by the shaders (trace.metal.h),
// bound whole as `constant StructureBlock&` (passes/bindings.h). Apple's
// way of putting an acceleration structure in an argument buffer: the host
// writes its gpuResourceID, 64 bits, where the shader reads an
// acceleration structure (Metal Shading Language, 2.17). InstanceStructure
// picks each side's type, a type alias rather than a macro (ES.30), as
// core/scene/scene_block.h does.
//
//   scene           the instance structure over every shape's triangle
//                   mesh, instance i shape i, each carrying the mask bits
//                   of the questions it answers (everything_bit,
//                   occluder_bit: contract 19), traced by the intersector
//                   in hardware, MNEE's seed among its rays
//                   (docs/research/2026-10-23-apple-silicon-design.md,
//                   section 6, change 1).
//   occluder_count  how many shapes carry occluder_bit: none, and a shadow
//                   ray or the seed is not traced.
//
// The constant address space: 16 bytes every thread of every tracing pass
// reads before its traces, the same for all of them, the reuse that space
// is for (metal/scene/scene_block.metal.h, WWDC16 606). Fields largest
// first (CACHE.5), padded to 16 so the size is one statement on both
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
#else
using InstanceStructure = std::uint64_t;  // the structure's gpuResourceID
#endif

// The instances' mask bits (contract 19): every shape carries
// everything_bit; a shape that is not a light carries occluder_bit too.
// A namespace-scope constant is in the shading language's constant address
// space (core/contracts/shared_layout.h, SERENITY_CONSTANT).
#if defined(__METAL_VERSION__)
constant constexpr uint32_t everything_bit = 1u;
constant constexpr uint32_t occluder_bit = 2u;
#else
inline constexpr uint32_t everything_bit = 1u;
inline constexpr uint32_t occluder_bit = 2u;
#endif

struct StructureBlock {
  InstanceStructure scene;
  uint32_t occluder_count;
  uint32_t padding;
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
    sizeof(StructureBlock) == 16,
    "StructureBlock must be the same 16 bytes on the host and in shaders");

// A triangle as a mesh's structure keeps it beside the hardware's own copy,
// Metal's per-primitive data (MTL4::AccelerationStructureTriangleGeometry
// Descriptor's primitiveDataBuffer): its three vertices in the mesh's
// space, read from a hit's intersection result, so a hit's point is the
// triangle's (trace.metal.h). 36 bytes, packed floats on both sides.
struct MeshTriangle {
  float v[3][3];
};

static_assert(sizeof(MeshTriangle) == 36,
              "MeshTriangle must be the same 36 bytes on the host and in "
              "shaders");

}  // namespace gpu
}  // namespace serenity
