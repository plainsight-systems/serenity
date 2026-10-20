#include "cuda/passes/features/features.h"

#include "cuda/device/error.h"
#include "serenity/optix_modules/features.h"

namespace serenity::cuda {

FeaturesPass::FeaturesPass(const OptixContext& context,
                           TracedProgram::Records records)
    : program_(context, optix_modules::features, "__raygen__features",
               records) {}

void FeaturesPass::record(CUstream stream, CUdeviceptr launch,
                          const FrameLaunch& frame) const {
  if (frame.albedo == 0 || frame.normal == 0 || frame.depth == 0 ||
      frame.motion == 0 || frame.feature_last_transforms == 0) {
    throw CudaError(
        "FeaturesPass: the frame lacks a feature image or the shapes as the "
        "frame before placed them");
  }
  program_.launch(stream, launch, {frame.frame.width, frame.frame.height});
}

}  // namespace serenity::cuda
