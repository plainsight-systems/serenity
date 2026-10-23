#pragma once

#include <optix_types.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "core/shapes/shape_meshes.h"
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
// The structure rays are traced against, built for the RTX 3090 Ti's RT
// cores, which intersect triangles and instance transforms in hardware
// and nothing else (docs/research/2026-10-20-nvidia-hardware-design.md,
// Phase 1):
//
//   - the meshes (shape_meshes.h): an icosphere a level the scene's spheres
//     use (mesh_levels.h) and the unit box, each a geometry structure of
//     triangles, built once at construction for fast tracing and
//     compacted (GPU.9: long-lived, built once; the compacted copy kept,
//     the build's memory let go);
//   - the occluders' boxes, a geometry structure of custom primitives,
//     the core's plan of them (core/shapes/structure_plan.h): MNEE's seed
//     walks them with its own intersection program, which needs every
//     sphere a segment enters (integrator/mnee_seed.cuh). None in a scene
//     of lights alone;
//   - the instance level over them, the frame's one traversable: each
//     shape an instance of its mesh, its id the shape's index, placed by
//     its transform (contract 10: the 3 x 4 row-major matrix OptiX's
//     instances take) after its mesh's own placement, the identity for a
//     sphere and the box's corners for a box; its mask the bits of the
//     questions it answers (structure_block.h: every shape, surfaces as
//     occluders too); and the seed's boxes, one instance under the seed's
//     bit alone. Principle 11's two questions are the mask bits of one
//     structure, as DXR's and OptiX's instances carry them
//     (logical-overview.md, "A mask needs an instance level").
//
// Each sphere's level is the scene's (Roles::levels, from mesh_levels.h's
// sphere_levels()): the structure builds what it is told and refuses a
// sphere without one.
//
// A frame where shapes move, update(), on the frame's stream, before any
// pass traces:
//
//   1. the moving shapes' instances placed by the frame's transforms, into
//      the slot's staging copy of the instances, uploaded
//      (cuda/device/frame_upload.h); and the seed's boxes, where they move,
//      placed and uploaded (shapes::place_moving);
//   2. the seed's structure rebuilt where it moves, then the instance level
//      rebuilt over every instance, into the same memory.
//
// Rebuilt, not refit: a refit keeps a tree's shape, whose quality decays as
// shapes move away from where it was built (CDSA.30), and principle 9 keeps
// nothing built over the lights past its frame. The meshes never move: a
// moving shape is its instance's transform. One copy of each structure, one
// scratch, not one a frame in flight: the stream orders frame n + 1's
// uploads and builds after frame n's traces (frame_upload.h).
//
// Made once, outside the frame loop: every buffer at construction, so a
// frame allocates nothing (MEM.9). Released with this object, after the
// frames that traced it (submission.h, Lifetime), before the OptiX
// context.
//
// Throws CudaError if there are no shapes, if an index in the roles is not
// a shape's or a list is not increasing, if a sphere has no level or the
// levels are not one a shape (I.5), or if OptiX cannot build a structure.
class SceneAcceleration {
 public:
  // Which shapes move and which are lights, each by index in increasing
  // order, and each shape's icosphere level, in shape order
  // (mesh_levels.h: no_level for a shape that is not a sphere). Named
  // (I.24), so the lists cannot pass in each other's places.
  struct Roles {
    std::span<const std::uint32_t> moving;
    std::span<const std::uint32_t> lights;
    std::span<const std::uint8_t> levels;
  };

  SceneAcceleration(const OptixContext& context, Submission& submission,
                    const shapes::Shapes& shapes, Roles roles);

  SceneAcceleration(const SceneAcceleration&) = delete;
  SceneAcceleration& operator=(const SceneAcceleration&) = delete;
  SceneAcceleration(SceneAcceleration&&) = delete;
  SceneAcceleration& operator=(SceneAcceleration&&) = delete;
  ~SceneAcceleration() = default;

  // Whether a frame must call update() before it traces.
  bool moves() const noexcept { return !moving_.empty(); }

  // Steps 1 and 2 for `frame`, from its `transforms`, one a shape. Called
  // once a frame where shapes move, between Submission::begin() and any
  // pass that traces; throws CudaError when nothing moves, or if
  // `transforms` is not one a shape.
  void update(const FrameSlot& frame,
              std::span<const contracts::Transform> transforms);

  // The structure a frame traces, for its launch parameters, as of the
  // last build.
  StructureBlock block() const noexcept;

  // The hit records, in record order (structure_block.h), for a pass's
  // pipeline: the triangles' group with no data, and the seed's group with
  // the occluders' table and the shapes' arrays as `scene` and
  // `transforms` place them.
  std::array<Pipeline::HitRecord, structure_records> hit_records(
      const gpu::SceneBlock& scene, CUdeviceptr transforms) const noexcept;

 private:
  // A mesh's geometry structure, compacted (above). Its vertices and
  // triangles are let go once it is built.
  struct Mesh {
    DeviceBuffer structure;
    OptixTraversableHandle handle = 0;
  };

  // The seed's boxes, as the core planned them, box i shape
  // plan.members[i]. Made whole by make_seeds() alone (C.41).
  struct Seeds {
    shapes::TreePlan plan;
    std::unique_ptr<FrameUpload> boxes;  // one a member
    DeviceBuffer structure;
    DeviceBuffer scratch;  // released after a still one's one build
    OptixTraversableHandle handle = 0;
  };

  // Builds every mesh on `meshes`, compacts them, waits, and keeps the
  // compacted copies: start-up work, settled here (submission.h).
  void make_meshes(Submission& submission,
                   std::span<const shapes::TriangleMesh> meshes);
  Seeds make_seeds(Submission& submission, shapes::TreePlan plan) const;
  // Records the seeds' build over their boxes, and the instance level's
  // over the instances, on `stream`.
  void build_seeds(Seeds& seeds, CUstream stream);
  void build_instances(CUstream stream);
  // Instance `shape`'s transform: the shape's placed after its mesh's.
  void place(OptixInstance& instance, std::uint32_t shape,
             const contracts::Transform& transform) const;

  OptixDeviceContext context_ = nullptr;
  shapes::Shapes shapes_;  // the records and the boxes, for place()
  std::vector<std::uint32_t> moving_;
  std::vector<Mesh> meshes_;
  std::vector<std::uint32_t> mesh_of_;  // a shape each: its index in meshes_
  std::optional<Seeds> seeds_;
  std::uint32_t occluder_count_ = 0;
  DeviceBuffer occluder_table_;  // a uint32 a seed box

  std::unique_ptr<FrameUpload> instances_;  // an OptixInstance each
  std::uint32_t instance_count_ = 0;
  DeviceBuffer instance_structure_;
  DeviceBuffer instance_scratch_;
  OptixTraversableHandle instance_handle_ = 0;
};

}  // namespace serenity::cuda
