#pragma once

#include <cuda.h>

#include "core/frame/extent.h"
#include "core/presentation/offline_renderer.h"
#include "cuda/device/buffer.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (the denoiser's feature images: contract 17).
//
// The four images the features pass writes, Metal's FeatureImages
// (metal/frame/feature_images.h, which gives what read() costs): albedo,
// normal and motion four half floats a pixel, depth one float, contract
// 17's formats, at the frame's size, rows tightly packed in device memory.
// Made when the frame's size is first known and remade when it changes,
// the frames in flight finished first.
//
// read() copies them back for the headless renderer's --features: a
// readback (GPU.1), not for a frame loop, after every committed frame; each
// image a LinearImage (contract 13), three floats a pixel, the halves
// widened exactly: albedo and normal their RGB; motion its (dx, dy, dz);
// depth its one value in all three.
//
// Throws CudaError if the memory cannot be made, or read() before the
// first prepare().
class FeatureImages {
 public:
  explicit FeatureImages(Submission& submission);

  FeatureImages(const FeatureImages&) = delete;
  FeatureImages& operator=(const FeatureImages&) = delete;
  FeatureImages(FeatureImages&&) = delete;
  FeatureImages& operator=(FeatureImages&&) = delete;
  ~FeatureImages() = default;

  // Ready for a frame of `size`: made, or remade, if not made at it.
  void prepare(frame::Extent size);

  // 0 before the first prepare().
  CUdeviceptr albedo() const noexcept { return albedo_.address(); }
  CUdeviceptr normal() const noexcept { return normal_.address(); }
  CUdeviceptr depth() const noexcept { return depth_.address(); }
  CUdeviceptr motion() const noexcept { return motion_.address(); }

  using Read = presentation::OfflineRenderer::Features;
  Read read() const;

 private:
  Submission& submission_;
  frame::Extent size_;
  DeviceBuffer albedo_;
  DeviceBuffer normal_;
  DeviceBuffer depth_;
  DeviceBuffer motion_;
};

}  // namespace serenity::cuda
