#include "cuda/passes/path/path.h"

#include "serenity/optix_modules/path.h"

namespace serenity::cuda {

PathPass::PathPass(const OptixContext& context, TracedProgram::Records records)
    : program_(context, optix_modules::path, "__raygen__path", records) {}

void PathPass::record(CUstream stream, CUdeviceptr launch,
                      frame::Extent size) const {
  program_.launch(stream, launch, size);
}

}  // namespace serenity::cuda
