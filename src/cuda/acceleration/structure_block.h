#pragma once

// Axis: Acceleration (on the GPU: the structures a frame traces, by mask).
//
// What a pass's programs are told to trace the scene (Metal's
// StructureBlock, metal/acceleration/structure_block.h): one geometry
// structure for each ray mask (contract 19, core/contracts/ray_mask.h), by
// its traversable handle, and the hit record each is traced with, whose
// data, a StructureRecord (below), is what the intersection program tests a
// box's shape with (cuda/device/pipeline.h). Written by the host
// (scene_acceleration.h) into a pass's launch parameters each frame, read
// by the programs (trace.cuh). Compiled by both the host compiler and nvcc;
// CUDA's C++ on both sides, so one type.
//
//   everything        every shape, primitive i shape i (shapes/primitive.h),
//                     traced with hit record everything_record, whose data
//                     is 0: no table.
//   occluders         the surfaces alone, in shape order, traced with hit
//                     record occluders_record, whose data is the address of
//                     the table naming each box's shape. With no light in
//                     the scene, everything's own structure, its table the
//                     identity; with no surface, no structure (handle 0)
//                     and occluder_count 0, and a ray carrying occluders
//                     finds nothing.
//   occluder_count    the table's entries.
//
// 24 bytes, the handles first (CACHE.5), read once by each trace.

#include <optix_types.h>

#include <cstdint>

#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"

namespace serenity::cuda {

// The hit records a pass's pipeline holds, in this order (pipeline.h).
inline constexpr unsigned everything_record = 0;
inline constexpr unsigned occluders_record = 1;
inline constexpr unsigned structure_records = 2;

struct StructureBlock {
  OptixTraversableHandle everything;
  OptixTraversableHandle occluders;
  std::uint32_t occluder_count;
  std::uint32_t padding;
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): the layout
    sizeof(StructureBlock) == 24,
    "StructureBlock must be the same 24 bytes on the host and in programs");

// What the intersection program reads from its structure's hit record:
// the table that names a box's shape, 0 where box i is shape i; and the
// shapes' records, boxes and this frame's transforms (the scene's block's,
// core/scene/scene_block.h, and cuda/scene/shape_transforms.h), each one
// device copy for the run, so the record is written once, when the pass's
// pipeline is made. In the record, not in a pass's launch parameters: the
// program is the same for every pass that traces. Addresses on the host,
// pointers in the programs (gpu::Array, scene_block.h); four 64-bit words,
// the pipeline's HitData.
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
