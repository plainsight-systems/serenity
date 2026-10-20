#pragma once

#include <cuda.h>

#include "core/scene/scene.h"
#include "core/scene/scene_block.h"
#include "cuda/device/buffer.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Scene content (on the GPU).
//
// The scene's arrays on the GPU, the core's list of them (core/scene/
// scene_arrays.h), and the scene's block (core/scene/scene_block.h), the
// address of each, which a pass's launch parameters name: Metal's
// SceneBuffers (metal/scene/scene_buffers.h) on a GPU whose memory is its
// own. The arrays are copied once, at construction, into one allocation,
// each at an offset aligned to 256 bytes (GPU.9: one allocation, not one
// each), and the block into one of its own; both copies are done before
// the constructor returns, so every frame reads them whole. An array with
// no elements has the address of a zeroed block at the allocation's start,
// shared by every empty one, as on Metal: no pointer a program reads is
// null, and none is read through an empty array's (scene_arrays.h).
//
// Static (CDSA.32: static data put once into the layout its consumer
// reads): what changes a frame, the shapes' transforms and the lights'
// glows, is not here (shape_transforms.h, light_glows.h). Read by the
// programs from global memory, through const pointers; Metal's are in its
// constant address space, which it measured faster (docs/research/
// 2026-10-10-scene-block.md). Where this machine reads them from is its
// tuning, after the port.
//
// Released with this object (R.1), after the frames that read it
// (submission.h, Lifetime). Throws CudaError if the memory cannot be made
// or filled. Not performance-sensitive: one copy, at start-up.
class SceneBuffers {
 public:
  SceneBuffers(Submission& submission, const scene::SceneDescription& scene);

  SceneBuffers(const SceneBuffers&) = delete;
  SceneBuffers& operator=(const SceneBuffers&) = delete;
  SceneBuffers(SceneBuffers&&) = delete;
  SceneBuffers& operator=(SceneBuffers&&) = delete;
  ~SceneBuffers() = default;

  // The address of each array, as the host sees it.
  const gpu::SceneBlock& block() const noexcept { return block_; }

  // The block on the GPU, which a pass's launch parameters name.
  CUdeviceptr block_address() const noexcept { return block_buffer_.address(); }

 private:
  DeviceBuffer arrays_;
  gpu::SceneBlock block_{};
  DeviceBuffer block_buffer_;
};

}  // namespace serenity::cuda
