#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/MTL4AccelerationStructure.hpp>  // not in Metal.hpp's umbrella
#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "core/contracts/camera.h"
#include "core/frame/extent.h"
#include "core/shapes/shape_meshes.h"
#include "core/shapes/shapes.h"
#include "core/shapes/structure_plan.h"
#include "metal/acceleration/structure_block.h"
#include "metal/device/device.h"
#include "metal/device/frame_array.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Acceleration.
//
// The structure rays are traced against, built for the M3's ray-tracing
// unit as Blender Cycles builds its scenes on the M3 and as Apple asks
// (docs/research/2026-10-23-apple-silicon-design.md, sections 5 and 6,
// change 1): opaque triangle geometry, which the hardware intersects
// itself and runs no intersection function for (Apple, "maximize opaque
// triangle tests"), each structure built once and compacted, under one
// top level; the still shapes concatenated into one structure, as Apple
// asks of instances that overlap ("concatenate these instances together
// into a single primitive acceleration structure", tech talk 111374), the
// marbles' pile being that; the moving shapes instanced, as Apple asks of
// objects that move. The same meshes at the same levels as the CUDA
// backend (core/shapes/shape_meshes.h, mesh_levels.h), which instances
// every shape, NVIDIA's practice on the RTX 3090 Ti (cuda/acceleration/
// scene_acceleration.h):
//
//   - the instanced shapes' meshes, every moving shape's and every box's:
//     an icosphere a level they use and the unit box, each a primitive
//     structure of opaque triangles in its own space, its triangles'
//     vertices kept beside it as Metal's per-primitive data
//     (structure_block.h, MeshTriangle), so a hit reads the triangle it
//     hit from its intersection result (trace.metal.h);
//   - the still structures: the still occluders' and the still lights'
//     spheres, each placed in the world by its transform and concatenated,
//     each triangle's per-primitive data its world vertices and its shape;
//     two, since a mask is an instance's. Not the boxes: a box's faces are
//     two triangles each, so placed in the world its far corners' rounding
//     is every point of the face's, which offset_ray() does not cover
//     where the face is large and the point near the origin (the ground
//     400 m across: tests/gpu/leave_test.cpp); a sphere's triangles are
//     small against it, its point's rounding its own coordinates'. The
//     scenes have a few boxes, so instancing them costs little;
//   - each built once at construction for fast intersection and compacted
//     (GPU.9: long-lived, built once; the compacted copy kept, the build's
//     memory let go);
//   - the occluders' boxes, a primitive structure of bounding boxes, the
//     core's plan of them (core/shapes/structure_plan.h), each box's shape
//     its per-primitive data: manifold next event estimation's seed
//     traverses it once a segment, its one intersection function visiting
//     every box the segment enters (integrator/mnee.metal.h,
//     seed_segment; mnee_seed.metal), as the CUDA backend's seed does;
//     every shape's in a scene with no light. A walk of nearest hits
//     through the instance structure, Blender Cycles' seed, measured 3.3
//     ms slower at 3456 x 2234 on the marbles, whose segments cross piles
//     of marbles, each hit a traversal from the top (docs/research/
//     2026-10-23-apple-silicon-design.md, section 6, change 1); Apple asks
//     for intersection functions "only on geometries that need them", and
//     a question every shape on a segment answers is one;
//   - the instance structure, the one every other ray traces: each still
//     structure an instance at the identity, its user id still_instance;
//     then each instanced shape an instance of its mesh, its user id the
//     shape's index, placed by its transform after its mesh's placement
//     (the identity for a sphere, the box's corners for a box);
//     each instance's mask the bits of the questions its shapes answer
//     (structure_block.h: every shape, surfaces as occluders too).
//     Principle 11's two questions are the mask bits of one structure, as
//     Cycles' and DXR's instances carry them (logical-overview.md, "A mask
//     needs an instance level").
//
// Each sphere's level is the scene's (Roles::levels, from the core's
// sphere_levels()): the structure builds what it is told and refuses a
// sphere without one.
//
// The earlier design here, one level of world-space boxes whose exact hits
// the shader decided through an intersection query, measured faster than
// instances on this machine for the brass scene's 4 shapes
// (docs/research/2026-10-09-acceleration-structure.md); Apple's M3 guidance
// since (the intersector over the query, opaque triangles, overlapping
// instances concatenated) and the
// marbles' 661 spheres are why it is replaced, and the frame time is what
// this design is measured by.
//
// A frame where shapes move, update(), at the start of the frame's encoder:
//
//   1. the moving shapes' instances placed by the frame's transforms into
//      the slot's instances (a FrameArray, frame_array.h), and the seed's
//      boxes, where they move (shapes::place_moving);
//   2. the seed's structure rebuilt where it moves, and the slot's
//      instance structure rebuilt over every instance;
//   3. one barrier from the acceleration-structure stage to the dispatch
//      stage, so every pass traces the finished structure (GPU.7).
//
// Rebuilt, not refit, as Apple advises for an instance structure: refit is
// for deforming meshes, and the one instance structure "usually only
// contains at most a few thousand objects", so a full rebuild each frame
// is fine (WWDC22 10105); a refit keeps a tree's shape, whose quality
// decays as shapes move away from where it was built (CDSA.30). The meshes
// never move: a moving shape is its instance's transform. A still scene's
// instance structure is built once, at construction. Each slot's
// structures were last used by the frame two submissions back, which
// Submission::begin() waited for, so nothing in flight reads what a frame
// writes, and frames still overlap.
//
// Made once, outside the frame loop: every buffer, structure and descriptor
// at construction, so a frame allocates nothing (MEM.9); per frame, one
// instance descriptor written per moving shape on the CPU, and one or two
// builds and one barrier on the GPU.
//
// Last frame's structure is not kept: every frame traces its own. ReSTIR
// PT's shift to last frame's pixel traces this frame's (metal/passes/
// restir_pt/restir_pt.metal.h, "What is approximate").
//
// Throws MetalError if there are no shapes, if an index in the roles is not
// a shape's or a list is not increasing, if a sphere has no level or the
// levels are not one a shape (I.5), or if the device cannot make a
// structure, its scratch memory or a buffer, or a start-up build fails on
// the GPU (submission.h).
class SceneAcceleration {
 public:
  // What the structures are told of the shapes beyond their geometry:
  // which move and which are lights, each by index in increasing order,
  // and each shape's mesh level (core/shapes/mesh_levels.h: no_level for a
  // shape that is not a sphere). Named fields, so the lists cannot pass in
  // each other's places (I.24).
  struct Roles {
    std::span<const std::uint32_t> moving;
    std::span<const std::uint32_t> lights;
    std::span<const std::uint8_t> levels;
  };

  // Builds the meshes and the structures over `shapes` at rest. A
  // structure none of whose shapes moves is built here; one with a moving
  // shape, in each slot by its frame's update(). Throws MetalError as
  // above (I.5).
  SceneAcceleration(const Device& device, Submission& submission,
                    const shapes::Shapes& shapes, Roles roles);

  SceneAcceleration(const SceneAcceleration&) = delete;
  SceneAcceleration& operator=(const SceneAcceleration&) = delete;
  SceneAcceleration(SceneAcceleration&&) = delete;
  SceneAcceleration& operator=(SceneAcceleration&&) = delete;
  ~SceneAcceleration() = default;

  // Records, into the frame's `encoder`, steps 1 to 3 for slot `slot`,
  // from the frame's `transforms`, one per shape. Called once a frame,
  // before any pass that traces, when shapes move, between
  // Submission::begin() returning the slot and the frame's commit; throws
  // MetalError when nothing moves, or if `transforms` is not the shape
  // count.
  void update(MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
              std::span<const contracts::Transform> transforms);

  // The structures frame slot `slot` traces, as a shader binds them: the
  // address of its StructureBlock (structure_block.h).
  MTL::GPUAddress address(std::uint32_t slot) const;

  // A mesh as the rasterizer draws it (passes/features/raster.metal): its
  // vertices and its triangles by index, drawn as built, so a fragment's
  // primitive id is its triangle's index in the mesh's per-primitive data,
  // `triangles`; whether its vertices are in the world (a still structure,
  // drawn once) or its own space (an instanced mesh, drawn once for each
  // of its `shape_count` entries from `first_shape` of the frame's
  // instance list, each placed by its shape's instance transform).
  struct Draw {
    MTL::GPUAddress vertices;
    MTL::GPUAddress indices;
    MTL::GPUAddress triangles;
    std::uint32_t index_count;  // three a triangle
    std::uint32_t first_shape;
    std::uint32_t shape_count;  // 0 for a still structure
    bool in_world;
  };

  // A frame's draws and their instance list, an entry an instanced shape
  // drawn (raster()): its mesh's index in the mesh block above
  // gpu::instance_mesh_shift, its shape below (structure_block.h), so the
  // visibility buffer names the triangle drawn (passes/features/
  // visibility.h).
  struct Raster {
    std::span<const Draw> draws;
    MTL::GPUAddress instances = 0;
  };

  // What the features' raster pass draws in slot `slot` from the frame's
  // `transforms` (one per shape) seen by `camera` in an image of `size`,
  // as raster engines choose a level of detail and cull to the view
  // (core/shapes/raster_levels.h): the still structures whole; each
  // instanced sphere in the view at its raster level this frame, a mesh
  // made for each level up to the tracer's finest (draw-only where the
  // tracer has none); each box. Valid until the slot's next call; writes
  // the slot's list, which the frame two submissions back read
  // (Submission::begin() waited for it), and allocates nothing (MEM.9).
  // Throws MetalError if `transforms` is not the shape count or `slot`
  // no frame slot.
  Raster raster(std::uint32_t slot,
                std::span<const contracts::Transform> transforms,
                const contracts::Camera& camera, frame::Extent size);

  // The meshes as a shader rebuilds a hit from its shape and triangle
  // (structure_block.h, MeshBlock): its GPU address.
  MTL::GPUAddress mesh_block() const;

 private:
  // A structure, its scratch and their residency.
  struct Built {
    NS::SharedPtr<MTL::Buffer> scratch;
    NS::SharedPtr<MTL::AccelerationStructure> structure;
    Resident structure_resident;
    Resident scratch_resident;  // when it is rebuilt each frame
  };

  // A mesh as the hardware traces it: its vertices, indices and triangles
  // (the per-primitive data) in one array set, and its compacted
  // structure.
  struct Mesh {
    std::unique_ptr<StaticArrays> data;
    NS::SharedPtr<MTL::AccelerationStructure> structure;
    Resident resident;
  };

  // A triangle set as a structure is built from it: its vertices, its
  // triangles by index, and each triangle's per-primitive data.
  struct MeshArrays {
    std::vector<contracts::Float3> vertices;
    std::vector<std::array<std::uint32_t, 3>> triangles;
    std::vector<gpu::MeshTriangle> data;
  };
  // The seed's boxes: the core's plan of the occluders, their boxes a
  // slot (one when none moves), each box's shape (its per-primitive data),
  // and a structure a slot (one when none moves).
  struct Seeds {
    shapes::TreePlan plan;
    std::unique_ptr<FrameArray> boxes;
    std::unique_ptr<StaticArrays> shapes;
    std::array<NS::SharedPtr<MTL4::PrimitiveAccelerationStructureDescriptor>,
               frames_in_flight>
        descriptors;
    std::array<Built, frames_in_flight> structures;
    bool moves() const { return plan.moves(); }
    const Built& in(std::uint32_t slot) const {
      return moves() ? structures.at(slot) : structures[0];
    }
  };
  // The core's meshes, each made once on the host while the structures
  // are built.
  struct MeshCache {
    std::array<std::optional<shapes::TriangleMesh>,
               shapes::finest_sphere_level + 1>
        levels;
    std::optional<shapes::TriangleMesh> box;
  };

  // Shape `shape`'s mesh in its own space, by its kind and `levels`, made
  // into `cache` once; throws MetalError if it is a sphere with no level.
  const shapes::TriangleMesh& mesh_for(std::uint32_t shape,
                                       std::span<const std::uint8_t> levels,
                                       MeshCache& cache) const;
  // Each of `made` as a structure, into meshes_ in order, built and
  // compacted in start-up submissions.
  void make_meshes(const Device& device, Submission& submission,
                   std::span<const MeshArrays> made);
  // The seed's boxes, their descriptors and structures; a still one's
  // scratch added to `build_only` for the start-up build.
  void make_seeds(const Device& device, Submission& submission,
                  shapes::TreePlan plan, MTL::ResidencySet* build_only);
  // Instance `instance`, shape `shape`'s, placed by `transform`.
  void place(std::uint32_t instance, std::uint32_t shape,
             const contracts::Transform& transform,
             std::span<MTL::IndirectAccelerationStructureInstanceDescriptor>
                 instances) const;

  // The shapes' records and boxes, which the instances' placements read.
  std::vector<shapes::ShapeRecord> records_;
  std::vector<shapes::BoxData> box_geometry_;
  std::vector<std::uint32_t> moving_;
  std::vector<Mesh> meshes_;
  // the shapes that are their own instances, in shape order, after the
  // still structures' instances; each one's mesh, as an index into
  // meshes_; each moving shape's instance, as moving_ orders them
  std::vector<std::uint32_t> instanced_;
  std::vector<std::uint32_t> shape_mesh_;
  std::vector<std::uint32_t> moving_instance_;
  // Each slot's instances (one copy when nothing moves), and its instance
  // structure's descriptor and structure.
  std::unique_ptr<FrameArray> instances_;
  std::array<NS::SharedPtr<MTL4::InstanceAccelerationStructureDescriptor>,
             frames_in_flight>
      tops_descriptors_;
  std::array<Built, frames_in_flight> tops_;
  bool moves_ = false;
  std::optional<Seeds> seeds_;  // none in a scene of lights alone
  // The rasterizer's: each mesh's draw, as built; each sphere level's mesh
  // up to raster_finest_ (meshes_ indices; the tracer's where it has one,
  // a draw-only one else) and the box's; each instanced shape's tracer
  // level; a slot's instance list (FrameArray) and draws, and the frame's
  // count of each mesh's entries, all sized at construction.
  std::vector<Draw> mesh_draws_;
  std::vector<std::uint32_t> raster_mesh_;
  std::optional<std::uint32_t> raster_box_;
  std::vector<std::uint8_t> trace_level_;
  std::unique_ptr<FrameArray> raster_instances_;
  std::array<std::vector<Draw>, frames_in_flight> raster_draws_;
  std::vector<std::uint32_t> mesh_counts_;
  std::vector<std::uint32_t> mesh_cursors_;
  std::vector<std::uint32_t> chosen_mesh_;
  // On the GPU: the meshes' records and each shape's mesh; then the mesh
  // block over them.
  std::unique_ptr<StaticArrays> mesh_tables_;
  std::unique_ptr<StaticArrays> mesh_block_;
  // one StructureBlock, one per slot when anything moves
  std::unique_ptr<FrameArray> blocks_;
};

}  // namespace serenity::metal
