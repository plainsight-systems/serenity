#pragma once

// Axis: Acceleration (on the GPU: the structure a frame traces, by mask).
//
// What a pass's programs are told to trace the scene: one instance level
// over the shapes' meshes (scene_acceleration.h,
// docs/research/2026-10-20-nvidia-hardware-design.md), by its traversable
// handle, each shape an instance whose id is the shape's index and whose
// mask bits answer the questions contract 19 asks (core/contracts/
// ray_mask.h), and one more instance, MNEE's seed's. Written by the host
// into a pass's launch parameters each frame, read by the programs
// (trace.cuh). Compiled by both the host compiler and nvcc; CUDA's C++ on
// both sides, so one type.
//
//   scene            the instance level: every shape, and the seed's boxes.
//   occluder_count   how many shapes are surfaces: 0 in a scene of lights
//                    alone, where a ray carrying occluders finds nothing
//                    and is not traced.
//
// 16 bytes, the handle first (CACHE.5), read once by each trace.

#include <optix_types.h>

#include <cstdint>

#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"

namespace serenity::cuda {

// The instances' mask bits (OptiX tests ray mask & instance mask, 8 bits):
// every shape carries everything_bit, a surface occluder_bit too (no light
// does), and the seed's instance seed_bit alone. A ray carries the one bit
// of its question, so the seed's boxes are met only by a seed's segment.
inline constexpr unsigned everything_bit = 1u;
inline constexpr unsigned occluder_bit = 2u;
inline constexpr unsigned seed_bit = 4u;

// The hit records a pass's pipeline holds, in this order (pipeline.h),
// picked by the instance's SBT offset, a trace's own being 0: the shapes'
// meshes, OptiX's built-in triangles with no program; and MNEE's seed's
// intersection program over the occluders' boxes (integrator/
// mnee_seed.cuh).
inline constexpr unsigned triangles_record = 0;
inline constexpr unsigned seed_record = 1;
inline constexpr unsigned structure_records = 2;

// The hit groups, in the order a pass's pipeline is given them, and the
// payload words a trace carries: the shape a seed skips, and a seed's
// address in two.
inline constexpr unsigned triangles_group = 0;
inline constexpr unsigned seed_group = 1;
inline constexpr unsigned trace_payload_values = 3;

struct StructureBlock {
  OptixTraversableHandle scene;
  std::uint32_t occluder_count;
  std::uint32_t padding;
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): the layout
    sizeof(StructureBlock) == 16,
    "StructureBlock must be the same 16 bytes on the host and in programs");

// What MNEE's seed program reads from its hit record: the table that
// names an occluders' box's shape, and the shapes' records, boxes and this
// frame's transforms (the scene's block's, core/scene/scene_block.h, and
// cuda/scene/shape_transforms.h), each one device copy for the run, so the
// record is written once, when the pass's pipeline is made. Addresses on
// the host, pointers in the programs (gpu::Array, scene_block.h); four
// 64-bit words, the pipeline's HitData. The triangles' record carries
// none: a mesh's hit names its shape by its instance.
struct StructureRecord {
  gpu::Array<uint32_t> table;
  gpu::Array<shapes::ShapeRecord> shapes;
  gpu::Array<shapes::BoxData> boxes;
  gpu::Array<contracts::Transform> transforms;
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): the layout
    sizeof(StructureRecord) == 32,
    "StructureRecord must be the same four addresses on the host and in "
    "programs");

}  // namespace serenity::cuda
