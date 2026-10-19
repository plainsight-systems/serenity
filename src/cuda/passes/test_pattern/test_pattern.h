#pragma once

#include <cuda.h>

#include "core/contracts/frame_constants.h"
#include "cuda/device/module.h"

namespace serenity::cuda {

// Axis: Pass.
//
// The test pattern, CUDA's: needs no scene, writes the target from the
// frame's constants alone (contract 7): red across, green down, blue
// pulsing (test_pattern.cu), what Metal's pass draws, its first frames on
// this backend. Its constants are the kernel's parameter, by value, which
// the driver copies at the launch: 32 bytes, no ring (GPU.7's in-flight
// copies are the driver's here).
//
// Throws CudaError if its module cannot be loaded or the launch fails.
class TestPatternPass {
 public:
  TestPatternPass();

  // Launches the pattern over `target`, an RGBA8 image of the constants'
  // width and height, on `stream`.
  void record(CUstream stream, const contracts::FrameConstants& frame,
              CUdeviceptr target) const;

 private:
  Module module_;
  CUfunction kernel_ = nullptr;
};

}  // namespace serenity::cuda
