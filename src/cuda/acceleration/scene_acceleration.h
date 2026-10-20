#pragma once

#include <optix_types.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "core/shapes/shapes.h"
#include "core/shapes/structure_plan.h"
#include "cuda/acceleration/structure_block.h"
#include "cuda/device/buffer.h"
#include "cuda/device/frame_upload.h"
#include "cuda/device/optix_context.h"
#include "cuda/device/pipeline.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Acceleration.
//
// The structures rays are traced against, Metal's SceneAcceleration
// (metal/acceleration/scene_acceleration.h, which gives the design and its
// measurements) as OptiX builds them (docs/research/2026-10-19-cuda-
// backend.md, "Phase 3's design"): one geometry structure of custom
// primitives a ray mask, each primitive a bounding box, the core's plan of
// which shapes each holds and where their boxes are (core/shapes/
// structure_plan.h). The exact hit inside a box is the shape kind's own
// test, the intersection program (trace.cuh). No instance level, as on
// Metal: the pipeline traces a single geometry structure (pipeline.h).
//
// A box is the core's shapes::Bounds, two packed triples of floats, which
// is OptixAabb's layout (asserted in the .cpp): the plan's boxes are handed
// to OptiX as they are.
//
// A still structure is built once, at construction, in a submission of its
// own that is waited for (GPU.8: whole before any frame traces it), and its
// build's scratch let go (GPU.9). A moving one is rebuilt every frame by
// update(), on the frame's stream, before any pass traces it:
//
//   1. the moving boxes, where the frame's transforms place their shapes
//      (shapes::place_moving), into the slot's staging copy of the boxes,
//      uploaded (cuda/device/frame_upload.h);
//   2. the structure rebuilt over every box, into the same memory, with
//      its own scratch.
//
// One structure, one scratch and one device copy of the boxes, not one a
// frame in flight: the stream orders frame n + 1's upload and build after
// frame n's traces (frame_upload.h). Rebuilt, not updated (OptiX's refit),
// as Metal rebuilds: a refit keeps the tree's shape, whose quality decays
// as shapes move away from where it was built (CDSA.30), and principle 9
// keeps nothing built over the lights past its frame.
//
// Build flags none, OptiX's defaults, as Metal builds with its own; no
// compaction. Choosing them is this machine's tuning, after the port.
//
// Made once, outside the frame loop: every buffer at construction, so a
// frame allocates nothing (MEM.9); per frame, one upload of the boxes and
// one build a moving structure, on the stream. Released with this object,
// after the frames that traced it (submission.h, Lifetime), before the
// OptiX context.
//
// Throws CudaError if there are no shapes, if an index in the roles is not
// a shape's or a list is not increasing (I.5), or if OptiX cannot build a
// structure.
class SceneAcceleration {
 public:
  // Which shapes move and which are lights, each by index in increasing
  // order (I.24: named, so the two lists cannot pass in each other's
  // places).
  struct Roles {
    std::span<const std::uint32_t> moving;
    std::span<const std::uint32_t> lights;
  };

  SceneAcceleration(const OptixContext& context, Submission& submission,
                    const shapes::Shapes& shapes, Roles roles);

  SceneAcceleration(const SceneAcceleration&) = delete;
  SceneAcceleration& operator=(const SceneAcceleration&) = delete;
  SceneAcceleration(SceneAcceleration&&) = delete;
  SceneAcceleration& operator=(SceneAcceleration&&) = delete;
  ~SceneAcceleration() = default;

  // Whether a frame must call update() before it traces.
  bool moves() const noexcept { return everything_.moves(); }

  // Steps 1 and 2 for `frame`, from its `transforms`, one a shape, for each
  // moving structure. Called once a frame where shapes move, between
  // Submission::begin() and any pass that traces; throws CudaError when
  // nothing moves, or if `transforms` is not one a shape.
  void update(const FrameSlot& frame,
              std::span<const contracts::Transform> transforms);

  // The structures a frame traces, for its launch parameters, as of the
  // last build.
  StructureBlock block() const noexcept;

  // The hit records, in record order (structure_block.h), for a pass's
  // pipeline: each its program's group and the shapes' arrays as `scene`
  // and `transforms` place them, with no table for everything and the
  // occluders' table for the occluders, the seed's among them.
  std::array<Pipeline::HitRecord, structure_records> hit_records(
      const gpu::SceneBlock& scene, CUdeviceptr transforms) const noexcept;

 private:
  // One structure over some of the shapes, as the core planned it, box i
  // shape plan.members[i]. Made whole by make_tree() alone (C.41).
  struct Tree {
    shapes::TreePlan plan;
    std::unique_ptr<FrameUpload> boxes;  // one a member
    DeviceBuffer structure;
    DeviceBuffer scratch;  // released after a still tree's one build
    OptixTraversableHandle handle = 0;

    bool moves() const noexcept { return plan.moves(); }
  };

  static Tree make_tree(const OptixContext& context, Submission& submission,
                        shapes::TreePlan plan);
  // Records `tree`'s build over its boxes on `stream`.
  void build(Tree& tree, CUstream stream) const;

  OptixDeviceContext context_ = nullptr;
  Tree everything_;
  // The occluders' own, when the scene has lights and surfaces both;
  // otherwise none, and the block says what a ray carrying them traces.
  std::optional<Tree> occluders_;
  bool no_surfaces_ = false;
  std::uint32_t occluder_count_ = 0;
  DeviceBuffer occluder_table_;  // a uint32 a box
};

}  // namespace serenity::cuda
