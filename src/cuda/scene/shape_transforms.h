#pragma once

#include <cuda.h>

#include <cstdint>
#include <span>

#include "core/contracts/transform.h"
#include "cuda/device/frame_upload.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Scene content (on the GPU: where each shape is).
//
// The shapes' transforms (contract 10, core/contracts/transform.h) as a
// frame places them, Metal's ShapeTransforms (metal/scene/
// shape_transforms.h, which says what reads them and why one value places
// the shape hit, the light lit and the box traced). This file says which
// array it is; how it reaches the GPU a frame, and when the CPU may write
// a slot's copy, is FrameUpload's (cuda/device/frame_upload.h). Every copy
// starts as the shapes at rest; in a scene where shapes move, the core
// rewrites the moving ones in the slot's copy (animation::animate) and the
// frame uploads it. A still scene uploads nothing after construction.
//
// Last frame's transforms, which ReSTIR PT's temporal reuse reads, arrive
// with it (Phase 4).
//
// Throws CudaError if there are no shapes or the memory cannot be made.
class ShapeTransforms {
 public:
  ShapeTransforms(Submission& submission,
                  std::span<const contracts::Transform> at_rest);

  ShapeTransforms(const ShapeTransforms&) = delete;
  ShapeTransforms& operator=(const ShapeTransforms&) = delete;
  ShapeTransforms(ShapeTransforms&&) = delete;
  ShapeTransforms& operator=(ShapeTransforms&&) = delete;
  ~ShapeTransforms() = default;

  // Frame slot `slot`'s transforms, to place the frame's shapes in, under
  // FrameUpload's rule for writing.
  std::span<contracts::Transform> transforms(std::uint32_t slot) {
    return array_.view<contracts::Transform>(slot);
  }

  // The frame's transforms, from its slot's copy, on its stream.
  void upload(const FrameSlot& frame) { array_.upload(frame); }

  // The transforms, as a program reads them.
  CUdeviceptr address() const noexcept { return array_.address(); }

 private:
  FrameUpload array_;
};

}  // namespace serenity::cuda
