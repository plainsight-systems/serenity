#pragma once

#include <Metal/Metal.hpp>

#include "core/scene/scene.h"
#include "core/scene/scene_block.h"
#include "metal/device/device.h"
#include "metal/device/resident.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Scene content (on the GPU).
//
// The scene's arrays on the GPU, the core's list of them (core/scene/
// scene_arrays.h), and the scene's block (core/scene/scene_block.h), the
// address of each, which a pass binds whole. How they are allocated,
// aligned and made resident is the GPU backend's (metal/device/
// static_arrays.h); an array with no elements has the address of a zeroed
// block there.
//
// Static: written once, at construction. The shapes' transforms, which
// change when shapes move, are not here: ShapeTransforms keeps them, a copy
// per frame in flight when they move (shape_transforms.h).
//
// Throws MetalError if the arrays cannot be put on the GPU (static_arrays.h).
class SceneBuffers {
 public:
  SceneBuffers(const Device& device, Submission& submission,
               const scene::SceneDescription& scene);

  SceneBuffers(const SceneBuffers&) = delete;
  SceneBuffers& operator=(const SceneBuffers&) = delete;
  SceneBuffers(SceneBuffers&&) = delete;
  SceneBuffers& operator=(SceneBuffers&&) = delete;
  ~SceneBuffers() = default;

  // The address of each array (core/scene/scene_block.h), as the host sees it.
  const gpu::SceneBlock& block() const noexcept { return block_; }

  // The block on the GPU, which a pass binds whole (core/scene/scene_block.h).
  MTL::GPUAddress block_address() const noexcept {
    return block_buffer_.address();
  }

 private:
  StaticArrays arrays_;
  gpu::SceneBlock block_{};
  ResidentBuffer block_buffer_;
};

}  // namespace serenity::metal
