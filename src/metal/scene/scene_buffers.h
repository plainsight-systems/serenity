#pragma once

#include <Metal/Metal.hpp>

#include "core/scene/scene.h"
#include "metal/scene/scene_block.h"
#include "metal/device/device.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Scene content (on the GPU).
//
// Which of the scene description's arrays (core/scene/scene.h) the GPU
// holds, and the scene's block (scene_block.h), the address of each, which
// a pass binds whole. This file and the block are the list of them and
// nothing more: how they are allocated, aligned and made resident is the
// GPU backend's (metal/device/static_arrays.h). A new kind's array is added
// here and to the block, as it is to the description, and no pass changes;
// a change to how arrays reach the GPU is made there. Falcor's Scene keeps
// the same list, in its scene's parameter block.
//
// The layouts are the core's shared ones, so the bytes are copied as they
// are: nothing is converted, and the shaders read exactly what the scene
// reader wrote. An array with no elements has the address of a zeroed
// block (static_arrays.h): a shader indexes it only through a record that
// names its kind, and no record names a kind the scene has none of.
//
// Static: written once, at construction. The shapes' transforms, which
// change when shapes move, are not here: ShapeTransforms keeps them, a copy
// per frame in flight when they move (shape_transforms.h).
//
// Throws MetalError if the arrays cannot be put on the GPU (static_arrays.h).
class SceneBuffers {
public:
    SceneBuffers(const Device& device, Submission& submission, const scene::SceneDescription& scene);

    SceneBuffers(const SceneBuffers&) = delete;
    SceneBuffers& operator=(const SceneBuffers&) = delete;
    SceneBuffers(SceneBuffers&&) = delete;
    SceneBuffers& operator=(SceneBuffers&&) = delete;
    ~SceneBuffers() = default;

    // The address of each array (scene_block.h), as the host sees it.
    const gpu::SceneBlock& block() const { return block_; }

    // The block on the GPU, which a pass binds whole (scene_block.h).
    MTL::GPUAddress block_address() const { return block_buffer_->gpuAddress(); }

private:
    StaticArrays arrays_;
    gpu::SceneBlock block_{};
    NS::SharedPtr<MTL::Buffer> block_buffer_;
    Resident resident_;
};

}  // namespace serenity::metal
