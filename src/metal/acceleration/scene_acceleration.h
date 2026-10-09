#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/scene/animate.h"
#include "core/shapes/shapes.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Acceleration.
//
// The structure rays are traced against, kept as Unreal keeps its ray
// tracing scene: each geometry once, in its own coordinates, built once; and
// every shape an instance of one, placed by its transform, in a top-level
// structure rebuilt each frame from the instances. A shape that moves
// changes its transform and nothing else: no geometry is ever rebuilt for
// it (Epic, "Ray Tracing Performance Guide": static meshes' bottom-level
// structures are built once, at load; the top level is rebuilt every
// frame, at a cost that grows with the instance count; only geometry that
// deforms rebuilds its own, and nothing here deforms).
//
// The geometries (shapes/primitive.h): the unit sphere, shared by every
// sphere of the scene, and each box. Each is a primitive structure of one
// axis-aligned bounding box, its object-space bounds (shapes::
// object_bounds), built at construction, in a submission of its own that is
// waited for, and never touched again (CDSA.29). Metal traces the boxes in
// hardware; the exact hit inside one is the shape kind's own object-space
// test, which the tracing loop calls for each candidate (trace.metal.h).
// Boxes, not triangles: a sphere's silhouette and normal are then exact,
// which glass magnifies. Metal has no sphere primitive.
//
// The top: an instance structure with one instance per shape, in the
// scene's order: instance i is shape i, its user ID i (shapes/primitive.h),
// its transform shape i's (shapes/transform.h, given row-major, as the core
// lays it out), its structure its geometry's. So a hit names its shape
// directly, with no table between.
//
// A still scene (core/scene/animate.h, moves()) has one top, built at
// construction with the geometries, and every frame traces it. A scene
// where shapes move has a top per frame in flight, each with its own
// instance descriptors and scratch. A frame:
//
//   1. update() copies each moving shape's transform, which the core placed
//      for the frame's time in the frame's transforms (metal/scene/
//      shape_transforms.h), into the slot's instance descriptor. Only the
//      moving shapes' descriptors are written; the still ones were written
//      at construction and never change.
//   2. It records the build of the slot's top over all the instances, at the
//      start of the frame's encoder,
//   3. and a barrier from the acceleration-structure stage to the dispatch
//      stage, so every pass traces the finished structure (GPU.7).
//
// Rebuilt, not refit: refitting keeps the tree's shape, whose quality decays
// as instances move away from where it was built, without bound for free
// flight; Epic rebuilds the top every frame; and logical-overview.md's
// principle 9 says nothing built over the lights outlives the frame. The
// slot's descriptors, top and scratch were last used by the frame two
// submissions back, which Submission::begin() waited for: nothing in flight
// reads what the frame writes, and frames still overlap.
//
// Two things this relies on of Metal, which tests/gpu show on this machine:
// that an intersection query hands each candidate the ray in its instance's
// object space, its direction transformed but not renormalized, so the
// object-space t is the world's (shapes/transform.h); and that the instance
// transforms are applied in the traversal hardware.
//
// Not taken from Unreal, and why: its instance culling drops instances far
// from or behind the camera, which changes the image, and a path tracer's
// reference must see what the camera cannot (principle 3); its residency
// and per-frame budgets for deforming geometry have nothing here to act on.
//
// Optimization: every per-frame descriptor buffer, top and scratch is made
// at construction, so a frame allocates nothing (MEM.9); per frame, 48 bytes
// copied per moving shape on the CPU, and one top build and one barrier on
// the GPU, whose cost grows with the shape count and is the whole of what
// a moving shape costs the structure. No geometry is compacted: there are
// a handful, of one box each. Measured on the M3 Max: (the implementation
// fills in the top's build time for the first moving scene's 4 shapes and
// for 4096.)
//
// Throws Error if there are no shapes, if the device cannot make a
// structure, its scratch memory or a buffer, or if the start-up build fails
// on the GPU (submission.h).
class SceneAcceleration {
public:
    // Builds a structure for each geometry of `shapes`, and the top over its
    // shapes at rest. `animation` says which shapes move: none, and the one
    // top is built here; some, and each slot's top is built by its frame's
    // update().
    SceneAcceleration(const Device& device, Submission& submission, const shapes::Shapes& shapes,
                      const scene::SceneAnimation& animation);

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
                std::span<const shapes::Transform> transforms) const;

    // The top frame slot `slot` traces, as a shader binds it.
    MTL::ResourceID resource(std::uint32_t slot) const;

private:
    struct Top {
        NS::SharedPtr<MTL::Buffer> instances;  // one descriptor per shape
        NS::SharedPtr<MTL::Buffer> scratch;
        NS::SharedPtr<MTL4::InstanceAccelerationStructureDescriptor> descriptor;
        NS::SharedPtr<MTL::AccelerationStructure> structure;
    };

    // Each geometry's structure, built once: the unit sphere's, then each box's.
    std::vector<NS::SharedPtr<MTL::AccelerationStructure>> geometries_;
    std::vector<std::uint32_t> moving_;  // the shapes that move, whose descriptors update() writes
    // One top for a still scene, in slot 0, which every frame traces; one per
    // slot when shapes move.
    std::array<Top, frames_in_flight> tops_;
};

}  // namespace serenity::metal
