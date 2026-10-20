#pragma once

#include <cuda.h>

#include "core/passes/tone_map.h"
#include "cuda/device/images.h"
#include "cuda/device/module.h"
#include "cuda/frame/frame_images.h"

namespace serenity::cuda {

// Axis: Pass (tone map).
//
// The frame's radiance as the look wants it shown, Metal's ToneMapPass
// (metal/passes/tone_map/tone_map.h, which gives its cost): what is computed
// is the core's (core/passes/tone_map.h, steps 1 to 6), with its constants;
// this pass is how CUDA computes it, in four kernels over the radiance
// image and the bloom pyramid (cuda/frame/frame_images.h), in 12 launches:
//
//   - steps 1 and 2 for B_0: the first down kernel, reading the radiance
//     image through its filtering texture, exposing and clamping (step 1)
//     each of its 13 reads, writing B_0 as half floats;
//   - step 2 for B_1 .. B_5: the down kernel, each reading the level before;
//   - step 3 for B_4 .. B_0: the up kernel, each reading the level below
//     and adding to its own, in place;
//   - steps 1 and 4 to 6: the finish kernel at the frame's size, reading
//     the radiance image and B_0, writing the target.
//
// Each launch reads what the one before wrote: the stream runs them in
// order, so none needs a barrier (Metal's eleven). The settings are the
// graph's, the same every frame, each launch's first parameter, by value.
//
// Throws CudaError if `settings` are out of range (core/passes/tone_map.h)
// or the module cannot be loaded.
class ToneMapPass {
 public:
  explicit ToneMapPass(const passes::ToneMap& settings);

  // Launches the pass on `stream`: from `radiance` through `images`'
  // pyramid into `target`, an RGBA8 image of the radiance's size. Throws
  // CudaError if a launch fails.
  void record(CUstream stream, const PitchedImage& radiance,
              const FrameImages& images, CUdeviceptr target) const;

 private:
  passes::ToneMap settings_;
  Module module_;
  CUfunction down_first_ = nullptr;  // step 2, from the exposed radiance
  CUfunction down_ = nullptr;        // step 2, from a level
  CUfunction up_ = nullptr;          // step 3
  CUfunction finish_ = nullptr;      // steps 4 to 6
};

}  // namespace serenity::cuda
