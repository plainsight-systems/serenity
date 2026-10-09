#pragma once

#include <cstdint>
#include <span>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/shapes/transform.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Scene content (on the GPU: where each shape is).
//
// The shapes' transforms (core/shapes/transform.h) as a frame places them:
// the one array of the scene that changes from frame to frame when shapes
// move (core/scene/animate.h). Every other array is still, and SceneBuffers
// holds it once (scene_buffers.h).
//
// Shaders read a shape's transform after a hit, to carry its object-space
// normal into the world, and a sphere light reads its shape's, which is
// where it is and how big (core/lights/sphere_light.h); the acceleration
// structure takes the moving shapes' transforms from here into its instance
// descriptors (metal/acceleration/scene_acceleration.h). So the shape a ray
// hits, the light lit from it and the instance traced are placed by the one
// value, which the core computed once.
//
// A still scene keeps one copy, which every frame binds. A scene where
// shapes move keeps one per frame in flight (submission.h's slots), so the
// CPU places the next frame's shapes while the GPU may still read the last
// frame's (GPU.7); every copy starts as the shapes at rest, and the core
// then rewrites only the moving ones (scene::animate), in the slot of the
// frame being recorded. transforms(slot) is that copy, for the CPU to
// write: only between Submission::begin() returning the slot and the
// frame's commit, when the frame that last used the slot has completed
// (begin() waited for it) and the GPU reads nothing in it. Metal makes the
// CPU's writes to shared memory visible to the GPU at commit.
//
// One buffer, the copies at a fixed stride, in shared memory, made resident
// through the submission for good.
//
// Throws Error if there are no shapes or the device cannot make the buffer.
//
// Cost: 48 bytes per shape per copy, 96 KB for a thousand fireflies in two
// copies; per frame, the moving shapes' 48-byte writes, nothing allocated.
class ShapeTransforms {
public:
    // `copies` is 1 for a still scene, frames_in_flight when shapes move.
    ShapeTransforms(const Device& device, Submission& submission, std::span<const shapes::Transform> at_rest,
                    std::uint32_t copies);

    ShapeTransforms(const ShapeTransforms&) = delete;
    ShapeTransforms& operator=(const ShapeTransforms&) = delete;
    ShapeTransforms(ShapeTransforms&&) = delete;
    ShapeTransforms& operator=(ShapeTransforms&&) = delete;
    ~ShapeTransforms() = default;

    // The copy frame slot `slot` reads, to place the frame's shapes in; see
    // above. Slot 0's for a still scene, whatever `slot`.
    std::span<shapes::Transform> transforms(std::uint32_t slot) const;

    // The copy frame slot `slot` reads, as a shader binds it.
    MTL::GPUAddress address(std::uint32_t slot) const;

private:
    NS::SharedPtr<MTL::Buffer> buffer_;
    std::size_t count_ = 0;   // shapes in each copy
    std::size_t stride_ = 0;  // bytes between copies
    std::uint32_t copies_ = 0;
};

}  // namespace serenity::metal
