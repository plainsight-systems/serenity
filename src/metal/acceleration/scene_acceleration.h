#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/MTL4AccelerationStructure.hpp>  // not in Metal.hpp's umbrella
#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/shapes/shape_meshes.h"
#include "core/shapes/shapes.h"
#include "core/shapes/structure_plan.h"
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
// triangle tests"), each mesh built once and compacted, each shape an
// instance of its mesh under one top level; the CUDA backend's design on
// the RTX 3090 Ti (cuda/acceleration/scene_acceleration.h), the same
// meshes at the same levels (core/shapes/shape_meshes.h, mesh_levels.h):
//
//   - the meshes: an icosphere a level the scene's spheres use and the
//     unit box, each a primitive structure of opaque triangles, its
//     triangles' vertices kept beside it as Metal's per-primitive data
//     (structure_block.h, MeshTriangle), so a hit reads the triangle it
//     hit from its intersection result (trace.metal.h); built once at
//     construction for fast intersection and compacted (GPU.9: long-lived,
//     built once; the compacted copy kept, the build's memory let go);
//   - the instance structure over the meshes, the one every ray traces,
//     manifold next event estimation's seed among them (a walk of nearest
//     hits, as Cycles' seed is: integrator/mnee.metal.h, seed_segment):
//     each shape an instance of its mesh, instance i shape i (its user id
//     too), placed by its transform after its mesh's own placement
//     (the identity for a sphere, the box's corners for a box), its mask
//     the bits of the questions it answers (structure_block.h: every
//     shape, surfaces as occluders too). Principle 11's two questions are
//     the mask bits of one structure, as Cycles' and DXR's instances carry
//     them (logical-overview.md, "A mask needs an instance level").
//
// Each sphere's level is the scene's (Roles::levels, from the core's
// sphere_levels()): the structure builds what it is told and refuses a
// sphere without one.
//
// The earlier design here, one level of world-space boxes whose exact hits
// the shader decided through an intersection query, measured faster than
// instances on this machine for the brass scene's 4 shapes
// (docs/research/2026-10-09-acceleration-structure.md); Apple's M3 guidance
// since (the intersector over the query, opaque triangles) and the
// marbles' 661 spheres are why it is replaced, and the frame time is what
// this design is measured by.
//
// A frame where shapes move, update(), at the start of the frame's encoder:
//
//   1. the moving shapes' instances placed by the frame's transforms into
//      the slot's instances (a FrameArray, frame_array.h);
//   2. the slot's instance structure rebuilt over every instance;
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
// instance descriptor written per moving shape on the CPU, and one build
// and one barrier on the GPU.
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

  // The meshes the levels name and the unit box, built and compacted in
  // start-up submissions; each shape's mesh, as an index into meshes_.
  void make_meshes(const Device& device, Submission& submission,
                   std::span<const std::uint8_t> levels);
  // Shape `shape`'s instance, placed by `transform`.
  void place(std::uint32_t shape, const contracts::Transform& transform,
             std::span<MTL::IndirectAccelerationStructureInstanceDescriptor>
                 instances) const;

  // The shapes' records and boxes, which the instances' placements read.
  std::vector<shapes::ShapeRecord> records_;
  std::vector<shapes::BoxData> box_geometry_;
  std::vector<std::uint32_t> moving_;
  std::vector<Mesh> meshes_;
  std::vector<std::uint32_t> shape_mesh_;  // meshes_ index, a shape
  // Each slot's instances (one copy when nothing moves), and its instance
  // structure's descriptor and structure.
  std::unique_ptr<FrameArray> instances_;
  std::array<NS::SharedPtr<MTL4::InstanceAccelerationStructureDescriptor>,
             frames_in_flight>
      tops_descriptors_;
  std::array<Built, frames_in_flight> tops_;
  bool moves_ = false;
  // one StructureBlock, one per slot when anything moves
  std::unique_ptr<FrameArray> blocks_;
};

}  // namespace serenity::metal
