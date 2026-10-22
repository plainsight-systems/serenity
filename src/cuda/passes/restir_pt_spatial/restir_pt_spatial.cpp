#include "cuda/passes/restir_pt_spatial/restir_pt_spatial.h"

#include "cuda/device/error.h"
#include "serenity/cuda_modules/restir_pt_spatial_kernels.h"
#include "serenity/optix_modules/restir_pt_spatial.h"

namespace serenity::cuda {

RestirPtSpatialPass::RestirPtSpatialPass(const OptixContext& context,
                                         TracedProgram::Records records)
    : sequence_(context, optix_modules::restir_pt_spatial,
                cuda_modules::restir_pt_spatial_kernels,
                ShiftReplay::Names{.pair = "restir_pt_spatial_pair",
                                   .reconnect = "__raygen__restir_pt_reconnect",
                                   .replay = "__raygen__restir_pt_replay",
                                   .merge = "restir_pt_spatial"},
                records) {}

void RestirPtSpatialPass::record(CUstream stream, CUdeviceptr launch,
                                 const FrameLaunch& frame) const {
  // The merge shades (restir_pt_spatial.h).
  if (frame.accumulated.rows == 0 || frame.non_finite == 0) {
    throw CudaError(
        "RestirPtSpatialPass: the frame lacks the accumulated image or its "
        "counter");
  }
  sequence_.record(stream, launch, frame);
}

}  // namespace serenity::cuda
