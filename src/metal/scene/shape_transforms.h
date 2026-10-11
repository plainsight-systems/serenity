#pragma once

#include <cstdint>
#include <span>

#include <Metal/Metal.hpp>

#include "core/contracts/transform.h"
#include "metal/device/device.h"
#include "metal/device/frame_array.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Scene content (on the GPU: where each shape is).
//
// The shapes' transforms (contract 10, core/contracts/transform.h) as a
// frame places them: the one array of the scene that changes from frame to
// frame when shapes move (core/animation/animate.h). Every other array is
// still, and SceneBuffers holds it once (scene_buffers.h). This file says
// which array it is, and nothing more: how it is allocated, made resident
// and copied per frame in flight, and when the CPU may write it, is the GPU
// backend's (metal/device/frame_array.h).
//
// Shaders read a shape's transform after a hit, to carry its object-space
// normal into the world, and a sphere light reads its shape's, which is
// where it is and how big (core/lights/sphere_light.h); the acceleration
// structure places the moving shapes' boxes by their transforms from here
// (metal/acceleration/scene_acceleration.h). So the shape a ray hits, the
// light lit from it and the box traced are placed by the one value, which
// the core computed once.
//
// A still scene keeps one copy; a scene where shapes move, one per frame in
// flight, every one starting as the shapes at rest, the core then rewriting
// only the moving ones in the copy of the frame being recorded
// (animation::animate).
//
// Last frame's transforms, for a graph with ReSTIR's temporal merge
// (metal/reuse/temporal.metal.h): the merge evaluates last frame's surface
// as last frame lit it, a light where last frame placed it, and the
// surfaces pass places a hit's point where last frame placed its shape. The
// renderer keeps a second ShapeTransforms for them, of this same class, and
// in each frame being recorded animates the moving shapes into its slot's
// copy at the previous frame's time (metal/frame/renderer.h). Not the
// previous slot's copy of this array: with two frames in flight, frame i
// would read frame i - 1's copy as last frame's, and frame i + 1, in the
// same slot as i - 1, rewrites it while frame i may still be running. A
// second array is the same copy-per-slot rule the
// first keeps (frame_array.h), and costs one more animation of the moving
// shapes a frame on the CPU and the array's size again. A still scene
// needs none: its one copy is last frame's too. Only a graph with a
// temporal merge has it.
//
// Throws MetalError if there are no shapes or the backend cannot make the array.
class ShapeTransforms {
public:
    // One copy of `at_rest`, or one per frame in flight when the shapes move.
    ShapeTransforms(const Device& device, Submission& submission, std::span<const contracts::Transform> at_rest,
                    FrameArray::Copies copies);

    ShapeTransforms(const ShapeTransforms&) = delete;
    ShapeTransforms& operator=(const ShapeTransforms&) = delete;
    ShapeTransforms(ShapeTransforms&&) = delete;
    ShapeTransforms& operator=(ShapeTransforms&&) = delete;
    ~ShapeTransforms() = default;

    // Frame slot `slot`'s transforms, to place the frame's shapes in, under
    // FrameArray's rule for writing (frame_array.h).
    std::span<contracts::Transform> transforms(std::uint32_t slot);

    // Frame slot `slot`'s transforms, as a shader binds them.
    MTL::GPUAddress address(std::uint32_t slot) const { return array_.address(slot); }

private:
    FrameArray array_;
};

}  // namespace serenity::metal
