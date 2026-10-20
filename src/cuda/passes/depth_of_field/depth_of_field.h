#pragma once

#include <cuda.h>

#include <cstdint>

#include "core/contracts/depth_of_field.h"
#include "core/frame/extent.h"
#include "cuda/device/module.h"
#include "cuda/frame/depth_of_field_images.h"

namespace serenity::cuda {

// What DepthOfFieldPass::record() reads, below (C.1: the frame's inputs to
// the pass, one object; C.2: plain data, no invariant of its own, which
// record() checks). The images by pointer (C.12: no reference member in a
// copyable type), null for a frame that has none, as Metal's
// FrameResources holds them.
struct DepthOfFieldInputs {
  // The radiance image, four floats a pixel at the output's size: row 0's
  // address and the bytes from a row to the next (PitchedImage).
  CUdeviceptr radiance = 0;
  std::uint64_t radiance_pitch = 0;
  frame::Extent size;  // the output's
  // The full-size depth: one float a pixel at the output's size, rows
  // tightly packed (contract 17).
  CUdeviceptr full_depth = 0;
  // The frame's, from passes::depth_of_field_constants(lens, size).
  contracts::DepthOfFieldConstants constants{};
  const DepthOfFieldImages* images = nullptr;
};

// Axis: Pass (depth of field: contract 21, core/contracts/depth_of_field.h).
//
// The scene's lens, applied to a denoised frame after the network, Metal's
// DepthOfFieldPass (metal/passes/depth_of_field/depth_of_field.h, which
// states every step and rule, what Jimenez's depth of field it keeps and
// what not, and gives its cost): its six kernels (depth_of_field.cu),
// translated line for line, each one launch on the frame's stream in
// Metal's order:
//
//   1. presort, at the half grid: each half pixel the 2 x 2 block's mean
//      and the signed circle of confusion of its nearest surface;
//   2. tiles, a thread a tile: each tile's largest near-field radius;
//   3. dilation, a thread a tile: the largest of the tiles that reach it;
//   4. the gather, at the half grid: scatter as gather over the 49-tap
//      disc into the far and near fields;
//   5. the median, at the half grid: a 3 x 3 median of both fields;
//   6. the composite, at the output's size, in place in the radiance image.
//
// Each launch reads what the one before wrote: the stream runs them in
// order, so none needs a barrier (GPU.8; Metal's five).
//
// What a frame gives it (DepthOfFieldInputs, above), what Metal's record()
// takes from FrameResources: the radiance image the denoise pass wrote at
// the output's size, which the presort reads and the composite reads and
// writes, by address and pitch, exactly (cuda/device/images.h,
// PitchedImage); the full-size depth (contract 17), one float a pixel,
// rows tightly packed; the pass's constants, by value, as the core fills
// them from the camera's lens (passes::depth_of_field_constants, core/
// passes/depth_of_field.h), each launch's first parameter; and its images
// (cuda/frame/depth_of_field_images.h), prepared at the output's size. The
// caller records nothing for a pinhole's frame (passes::blurs), whose
// image is the network's own, as Metal's renderer does.
//
// In place, the composite reads and writes each pixel of the radiance
// image by the same thread, after the presort, four launches before it,
// has read the image whole: so no full-size image of its own (GPU.9, as
// Metal's).
//
// Cost: Metal's (its header), the same reads and writes a pixel, some 230
// bytes, most of them cached; the tiles and dilation a thread a tile.
//
// Throws CudaError if the module cannot be loaded.
class DepthOfFieldPass {
 public:
  DepthOfFieldPass();

  // Launches the pass on `stream`. Throws CudaError if `inputs` lack the
  // radiance image, the full-size depth or the images (I.10, E.2: the
  // pass cannot do its task), as Metal's throws MetalError; if the images
  // or the constants are not at the output's size (a launch past an
  // image's edge would fault the context, losing every frame after it);
  // or if a launch fails.
  void record(CUstream stream, const DepthOfFieldInputs& inputs) const;

 private:
  Module module_;
  CUfunction presort_ = nullptr;
  CUfunction tiles_ = nullptr;
  CUfunction dilate_ = nullptr;
  CUfunction gather_ = nullptr;
  CUfunction median_ = nullptr;
  CUfunction composite_ = nullptr;
};

}  // namespace serenity::cuda
