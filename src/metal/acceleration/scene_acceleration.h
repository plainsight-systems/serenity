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
// than everything it saves: measured, with the variants tried and the
// triangle spheres' look, in docs/research/2026-10-09-acceleration-structure.md.
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
// Last frame's structure, kept one frame longer, for a graph with ReSTIR's
// temporal merge (Keep::last_too; metal/reuse/temporal.metal.h): its Z test
// asks whether a sample was visible from last frame's surface as last frame
// saw it, so its shadow ray is traced against the structure last frame
// traced, not this one (RTXDI's previous-frame BVH, which it falls back
// from when an application has none, at the cost of bias; here it is always
// there). Only for such a graph; every other keeps what it has now.
//
//   - A still scene: its one structure is last frame's too. Nothing more.
//   - A scene where shapes move: frames_in_flight + 1 structures, three,
//     not one per slot. Frame i builds structure i mod 3 (its index's, not
//     its slot's: the history's frames are consecutive by index,
//     core/frame/history.h, while a submission that is not a frame, a
//     readback, takes a slot) and traces it, and traces structure (i - 1)
//     mod 3 as last frame's. With one per slot, frame i + 1 would rebuild
//     the structure frame i still reads as last frame's while frame i may
//     be running. With three, the structure frame i + 1 rebuilds was last
//     traced by frames i - 1 and i - 2, and before recording its build the
//     frame waits for the last submission that traced it
//     (Submission::wait_until_complete), which begin() has already waited
//     for whenever every submission is a frame, so the wait returns at
//     once; it is there so that one that is not cannot make the ring
//     unsafe (GPU.7: a resource's lifetime tied to the frames that read
//     it; MEM.4). The boxes stay one copy per slot: a build reads its
//     frame's slot's boxes, which the CPU writes under FrameArray's rule.
//
// What is new is holding one more structure and its scratch, not building
// one: the build is the one each frame already pays. A structure here is
// one level of the shapes' placed boxes, its size linear in the shape
// count; the third adds one more of them and its scratch, made at
// construction (MEM.9, GPU.9).
//
// Throws MetalError if there are no shapes, if the device cannot make a
// structure, its scratch memory or a buffer, or if the start-up build fails
// on the GPU (submission.h).
class SceneAcceleration {
public:
    // Whether last frame's structure is kept for this frame to trace too.
    // A type, not a bool (I.4).
    enum class Keep { this_frame, last_too };

    // As the constructor below, keeping last frame's structure when `keep`
    // says (above). Declared for review; the constructor below becomes this
    // one with Keep::this_frame, building what it builds now.
    SceneAcceleration(const Device& device, Submission& submission, const shapes::Shapes& shapes,
                      std::span<const std::uint32_t> moving, Keep keep);

    // Builds the structure over `shapes` at rest. `moving` are the indices
    // of the shapes that move, in increasing order: none, and the one
    // structure is built here; some, and each slot's is built by its frame's
    // update(). Throws MetalError if an index is not a shape's.
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
    // MetalError when nothing moves, or if `transforms` is not the shape count.
    void update(MTL4::ComputeCommandEncoder* encoder, std::uint32_t slot,
                std::span<const contracts::Transform> transforms);

    // The structure frame slot `slot` traces, as a shader binds it.
    MTL::ResourceID resource(std::uint32_t slot) const;

    // With Keep::last_too: records, into the frame's `encoder`, steps 1 to 3
    // for frame `index`, into structure index mod 3, from the boxes of
    // `begun`'s slot and the frame's `transforms`, first waiting through
    // `submission` for the last submission that traced that structure; and
    // records that `begun`, this frame's submission, now traces it and the
    // one before. The slot and the sequence come together, as
    // Submission::begin() gave them, so neither can be passed for the other
    // or for the frame's index (I.24). Called as update() is, in its place.
    // Throws MetalError when the structure is not kept, when nothing moves,
    // or if `transforms` is not the shape count.
    void update_kept(MTL4::ComputeCommandEncoder* encoder, Submission& submission, const FrameSlot& begun,
                     std::uint64_t index, std::span<const contracts::Transform> transforms);

    // With Keep::last_too: the structures frame `index` traces, its own and
    // last frame's, as a shader binds them; for a still scene, the one
    // structure twice. Throws MetalError when the structure is not kept.
    struct Traced {
        MTL::ResourceID current;
        MTL::ResourceID last;
    };
    Traced traced(std::uint64_t index) const;

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
    // one per slot when shapes move. With Keep::last_too and shapes moving,
    // the implementation makes this frames_in_flight + 1, indexed by frame
    // index (above), each with the sequence of the last submission that
    // traced it.
    std::array<Built, frames_in_flight> structures_;
};

}  // namespace serenity::metal
