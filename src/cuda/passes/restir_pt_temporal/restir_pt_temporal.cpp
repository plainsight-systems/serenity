#include "cuda/passes/restir_pt_temporal/restir_pt_temporal.h"

#include "serenity/cuda_modules/restir_pt_temporal_kernels.h"
#include "serenity/optix_modules/restir_pt_temporal.h"

namespace serenity::cuda {

RestirPtTemporalPass::RestirPtTemporalPass(const OptixContext& context,
                                           TracedProgram::Records records)
    : sequence_(context, optix_modules::restir_pt_temporal,
                cuda_modules::restir_pt_temporal_kernels,
                ShiftReplay::Names{
                    .pair = "restir_pt_temporal_pair",
                    .reconnect = "__raygen__restir_pt_temporal_reconnect",
                    .replay = "__raygen__restir_pt_temporal_replay",
                    .merge = "restir_pt_temporal"},
                records) {}

}  // namespace serenity::cuda
