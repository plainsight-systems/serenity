#include "cuda/passes/test_pattern/test_pattern.h"

#include <array>

#include "serenity/cuda_modules/test_pattern.h"

namespace serenity::cuda {

TestPatternPass::TestPatternPass()
    : module_(serenity::cuda_modules::test_pattern),
      kernel_(module_.kernel("test_pattern")) {}

void TestPatternPass::record(CUstream stream,
                             const contracts::FrameConstants& frame,
                             CUdeviceptr target) const {
  contracts::FrameConstants constants = frame;
  CUdeviceptr image = target;
  std::array<void*, 2> arguments{&constants, &image};
  launch_per_pixel(stream, kernel_, {frame.width, frame.height}, arguments);
}

}  // namespace serenity::cuda
