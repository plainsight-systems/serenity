#include "cuda/passes/restir_pt_spatial/restir_pt_spatial.h"

#include "serenity/cuda_modules/restir_pt_spatial_kernels.h"
#include "serenity/optix_modules/restir_pt_spatial.h"

namespace serenity::cuda {

RestirPtSpatialPass::RestirPtSpatialPass(const OptixContext& context,
                                         TracedProgram::Records records)
    : sequence_(context, optix_modules::restir_pt_spatial,
                cuda_modules::restir_pt_spatial_kernels,
                ShiftReplay::Names{.shift = "__raygen__restir_pt_shift",
                                   .replay = "__raygen__restir_pt_replay",
                                   .merge = "restir_pt_spatial"},
                records) {}

}  // namespace serenity::cuda
