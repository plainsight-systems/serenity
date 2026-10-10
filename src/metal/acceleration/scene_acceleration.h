#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/MTL4AccelerationStructure.hpp>  // not in Metal.hpp's umbrella
#include <Metal/Metal.hpp>

#include "core/shapes/shapes.h"
#include "metal/device/device.h"
#include "metal/device/frame_array.h"
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
// than everything it saves. Measured on the M3 Max, the path tracer on the
// still brass scene at 3456 x 2234 (docs/research/
// 2026-10-09-acceleration-structure.md):
//
//   one level, exact spheres, world-space tests (main before)  7.8 ms
//   this: one level, exact spheres, transforms, object space    8.2 ms
//   instances of triangle spheres, 5,120 to 81,920 each    11.5 - 12.5 ms
//   instances of exact spheres, one per shape                  18.1 ms
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
// the GPU, whose cost grows with the shape count. Measured on the M3 Max: the
// wandering brass scene's frame takes 8.28 ms against the still scene's
// 8.17, the placing and the build together 0.11 ms; a build alone, in a
// command buffer of its own, 68 us over 4 shapes and 340 us over 4096.
//
// Throws Error if there are no shapes, if the device cannot make a
// structure, its scratch memory or a buffer, or if the start-up build fails
// on the GPU (submission.h).
class SceneAcceleration {
public:
    // Builds the structure over `shapes` at rest. `moving` are the indices
    // of the shapes that move, in increasing order: none, and the one
    // structure is built here; some, and each slot's is built by its frame's
    // update(). Throws Error if an index is not a shape's.
    SceneAcceleration(const Device& device, Submission& submission, const shapes::Shapes& shapes,
                      std::span<const std::uint32_t> moving);

    SceneAcceleration(const SceneAcceleration&) = delete;
    SceneAcceleration& operator=(const SceneAcceleration&) = delete;
    SceneAcceleration(SceneAcceleration&&) = delete;
    SceneAcceleration& operator=(SceneAcceleration&&) = delete;
    ~SceneAcceleration() = default;

    // Records, into the frame's `encoder`, steps 1 to 3 for slot `slot`,
    // from the frame's `transforms`, one per shape. Called once a frame,
    // before any pass that traces, when shapes move, between
    // Submission::begin() returning the slot and the frame's commit; throws
    // Error when nothing moves, or if `transforms` is not the shape count.
    void update(MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
                std::span<const contracts::Transform> transforms) const;

    // The structure frame slot `slot` traces, as a shader binds it.
    MTL::ResourceID resource(std::uint32_t slot) const;

private:
    struct Built {
        NS::SharedPtr<MTL::Buffer> scratch;
        NS::SharedPtr<MTL4::PrimitiveAccelerationStructureDescriptor> descriptor;  // over the slot's boxes
        NS::SharedPtr<MTL::AccelerationStructure> structure;
        Resident structure_resident;
        Resident scratch_resident;  // when shapes move: each frame's build writes the scratch
    };

    std::vector<std::uint32_t> moving_;          // the shapes that move, whose boxes update() writes
    std::vector<shapes::Bounds> moving_bounds_;  // each one's object-space bounds, in moving_'s order
    std::unique_ptr<FrameArray> boxes_;          // one box per shape, per structure
    // One structure for a still scene, in slot 0, which every frame traces;
    // one per slot when shapes move.
    std::array<Built, frames_in_flight> structures_;
};

}  // namespace serenity::metal
