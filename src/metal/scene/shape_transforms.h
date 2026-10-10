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
// Throws Error if there are no shapes or the backend cannot make the array.
class ShapeTransforms {
public:
    // One copy of `at_rest` when `moves` is false; one per frame in flight
    // when it is true.
    ShapeTransforms(const Device& device, Submission& submission, std::span<const contracts::Transform> at_rest,
                    bool moves);

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
