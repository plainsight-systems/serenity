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

#include "core/shapes/shapes.h"
#include "metal/device/device.h"
#include "metal/device/frame_array.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Acceleration.
//
// The structure rays are traced against: one primitive acceleration
// structure of axis-aligned bounding boxes, one per shape, in the scene's
// order, so primitive i is shape i (shapes/primitive.h). Each box is the
// shape's geometry's object-space bounds (shapes::object_bounds) placed in
// the world by its transform (contract 10, shapes::world_bounds), so the
// structure names no shape kind. Metal traces the boxes in hardware; the
// exact hit inside one is the shape kind's own test, in its object space,
// which the tracing loop calls for each candidate (trace.metal.h). Boxes,
// not triangles: a sphere's silhouette and normal are then exact, which
// glass magnifies, and the reference must be exact. Metal has no sphere
// primitive.
//
// How it compares with Unreal's ray tracing scene, which builds each mesh's
// bottom-level structure once and rebuilds a top level of instances every
// frame (Epic, "Ray Tracing Performance Guide"). A mesh is thousands of
// triangles, worth building once and placing by a transform; a shape here is
// one box, so its bottom level would hold nothing worth keeping. This one
// level is Unreal's top level with each instance's transform folded into its
// box: rebuilt each frame when shapes move, as Unreal rebuilds its top, at a
// cost that grows with the shape count as Unreal's grows with the instances.
// What it leaves out is the instance level each ray would otherwise cross,
// which on this hardware, for boxes whose hits the shader decides, costs more
// than everything it saves: measured, with the variants tried and the
// triangle spheres' look, in
// docs/research/2026-10-09-acceleration-structure.md.
//
// When a shape arrives that is a mesh (a table of triangles), it is a
// structure of its own, built once, under an instance: a second level for
// that, measured again then.
//
// A still scene (no shape moves) has one structure, built once, at
// construction (CDSA.29: a structure with no mutations pays for its build
// once), in a submission of its own that is waited for, so it is complete
// before any frame traces it (GPU.8); its build's scratch is then let go
// (GPU.9). A scene where shapes move has one per frame
// in flight, each with its own boxes and scratch; the boxes are a FrameArray
// (metal/device/frame_array.h), whose rule says when the CPU may write a
// slot's. Which shapes move it is told as their indices, nothing more of the
// scene's animation (core/animation/animate.h). A frame:
//
//   1. update() writes each moving shape's box, its object-space bounds
//      placed by the transform the core placed for the frame's time
//      (metal/scene/shape_transforms.h), into the slot's boxes. Only the
//      moving shapes' boxes are written; the still ones were written at
//      construction and never change.
//   2. It records the build of the slot's structure over every box, at the
//      start of the frame's encoder,
//   3. and a barrier from the acceleration-structure stage to the dispatch
//      stage, so every pass traces the finished structure (GPU.7).
//
// Rebuilt, not refit: a refit keeps the tree's shape, whose quality decays
// as shapes move away from where it was built, without bound for free
// flight (CDSA.30); and logical-overview.md's principle 9 says nothing built
// over the lights outlives the frame. The slot's boxes, structure and
// scratch were last used by the frame two submissions back, which
// Submission::begin() waited for: nothing in flight reads what the frame
// writes, and frames still overlap.
//
// Optimization: every per-frame buffer, structure and build descriptor is
// made at construction, so a frame allocates nothing (MEM.9); per frame, one
// 24-byte box per moving shape on the CPU, and one build and one barrier on
// the GPU, whose cost grows with the shape count. What the placing and the
// build cost a frame, and a build alone by shape count, is measured in the
// same note.
//
// Last frame's structure is not kept: every frame traces its own. ReSTIR
// PT's shift to last frame's pixel traces this frame's (metal/passes/
// restir_pt/restir_pt.metal.h, "What is approximate").
//
// Ray masks (contract 19, core/contracts/ray_mask.h). A ray carries a mask,
// everything or the occluders, and finds the shapes of its mask's classes
// alone, as it would have found them unmasked. Metal tests masks only
// against instances (MTLAccelerationStructureInstanceDescriptor's mask,
// tested by an intersection query over an instance structure; a primitive
// structure's query takes none), as DXR and Vulkan do; this level has no
// instances, for the cost measured in the note above. So each mask is a
// structure of its own, over the shapes of its classes, and a ray traces
// the one its mask names (trace.metal.h):
//
//   everything   every shape, as before masks: the structure above, built
//                the same way over the same boxes, so a ray that carries
//                everything traces exactly what it traced (the image of
//                every such ray is unchanged, bit for bit).
//   occluders    the surfaces alone, the shapes that are not lights (which
//                the renderer hands over as their indices, learned from the
//                scene's answer to which shapes are lights, contract 3, as
//                it hands over the ones that move), in shape order, its
//                primitive i named by a table of their shapes. Its own
//                boxes and builds, kept by the same rule: built once at
//                construction when none of its shapes moves, as in the
//                marbles, whose movers are all fireflies; per slot by
//                update() when one does. In the marbles: 37 boxes where
//                everything has 653.
//
// Split by what the rays need and, as it happens, by update rate (CDSA.30:
// the large still part built once, the small moving part rebuilt), but
// unlike a two-level structure each mask's rays cross one level, so no ray
// pays an instance level to skip the fireflies. The alternative, Metal's
// masks themselves: one bottom level per class under a top level of two
// instances, each carrying its class's bit, the ray's mask passed to the
// query (Step 3's layout). Measured against this on the marbles
// (docs/research/2026-10-17-milestone-4-plan.md, Step 2): the same image,
// bit for bit, and slower than no masks at all, since every ray, not only
// the visibility rays, pays the instance level. With no
// light, occluders is everything itself, its table the identity, and
// nothing more is built; with no surface, nothing for it to hold, and the
// block says so (structure_block.h).
//
// What the shaders bind is one block a frame slot (structure_block.h): each
// structure's resource ID and the occluders' table. Written at
// construction, per slot when anything moves, and never again: each slot's
// structures are the same objects every frame, rebuilt in place.
//
// Cost: the occluders' structure, its scratch and boxes beside
// everything's (bytes, at construction, MEM.9), and, when it moves, one
// more build a frame in the same encoder before the one barrier (GPU.8);
// a table of one uint32 a surface and 32 bytes a slot. Per ray: one load of
// the block, the same for every thread, and for the occluders one table
// load per candidate the hardware returns.
//
// Throws MetalError if there are no shapes, if the device cannot make a
// structure, its scratch memory or a buffer, or if the start-up build fails
// on the GPU (submission.h).
class SceneAcceleration {
 public:
  // What the structures are told of the shapes beyond their geometry:
  // which move and which are lights, each by index in increasing order.
  // Named fields, so the two lists cannot pass in each other's places
  // (I.24).
  struct Roles {
    std::span<const std::uint32_t> moving;
    std::span<const std::uint32_t> lights;
  };

  // Builds the structures over `shapes` at rest. A structure none of
  // whose shapes moves is built here; one with a moving shape, in each
  // slot by its frame's update(). Throws MetalError if an index in `roles`
  // is not a shape's, or a list is not in increasing order (I.5).
  SceneAcceleration(const Device& device, Submission& submission,
                    const shapes::Shapes& shapes, Roles roles);

  SceneAcceleration(const SceneAcceleration&) = delete;
  SceneAcceleration& operator=(const SceneAcceleration&) = delete;
  SceneAcceleration(SceneAcceleration&&) = delete;
  SceneAcceleration& operator=(SceneAcceleration&&) = delete;
  ~SceneAcceleration() = default;

  // Records, into the frame's `encoder`, steps 1 to 3 for slot `slot`,
  // from the frame's `transforms`, one per shape: for each structure with
  // a moving shape, its moving boxes and its build; then the one barrier.
  // Called once a frame, before any pass that traces, when shapes move,
  // between Submission::begin() returning the slot and the frame's
  // commit; throws MetalError when nothing moves, or if `transforms` is
  // not the shape count.
  void update(MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
              std::span<const contracts::Transform> transforms);

  // The structures frame slot `slot` traces, as a shader binds them: the
  // address of its StructureBlock (structure_block.h).
  MTL::GPUAddress address(std::uint32_t slot) const;

 private:
  struct Built {
    NS::SharedPtr<MTL::Buffer> scratch;
    NS::SharedPtr<MTL4::PrimitiveAccelerationStructureDescriptor>
        descriptor;  // over the slot's boxes
    NS::SharedPtr<MTL::AccelerationStructure> structure;
    Resident structure_resident;
    Resident scratch_resident;  // when shapes move: each frame's build writes
                                // the scratch
  };

  // One structure over some of the shapes, its box i shape members[i]
  // (for everything, every shape, so box i is shape i): one, in entry 0,
  // which every frame traces, when none of them moves; one per slot,
  // indexed by slot, when one does. Made whole by make_tree() alone
  // (C.41).
  struct Tree {
    std::vector<std::uint32_t>
        moving_boxes;  // the boxes of its moving shapes, which update() writes
    std::vector<std::uint32_t>
        moving_shapes;  // their shapes, whose transforms place them
    std::vector<shapes::Bounds>
        moving_bounds;  // their object-space bounds, in the same order
    std::unique_ptr<FrameArray> boxes;  // one box per member, per structure
    std::array<Built, frames_in_flight> structures;

    bool moves() const { return !moving_boxes.empty(); }
    // The structure slot `slot` traces.
    const Built& in(std::uint32_t slot) const {
      return moves() ? structures[slot] : structures[0];
    }
  };

  // Makes a tree over `members`, the shapes it holds in increasing order,
  // of which `moving` (increasing) move: its boxes at rest, descriptors,
  // structures and scratch, resident; a still tree's scratch is added to
  // `build_only` instead, for the constructor's one start-up build.
  static Tree make_tree(const Device& device, Submission& submission,
                        const shapes::Shapes& shapes,
                        std::span<const std::uint32_t> members,
                        std::span<const std::uint32_t> moving,
                        MTL::ResidencySet* build_only);

  Tree everything_;
  // The occluders' own, when the scene has lights and surfaces both;
  // otherwise none, and the block names everything's (structure_block.h).
  std::optional<Tree> occluders_;
  std::unique_ptr<StaticArrays>
      occluder_table_;  // the occluders' shapes, a uint32 a box
  std::unique_ptr<FrameArray>
      blocks_;  // one StructureBlock, one per slot when anything moves
};

}  // namespace serenity::metal
