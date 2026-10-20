#include "cuda/passes/restir_pt_initial/restir_pt_initial.h"

#include <array>

#include "core/contracts/light_tile.h"
#include "cuda/device/error.h"
#include "serenity/cuda_modules/restir_pt_initial_kernels.h"
#include "serenity/optix_modules/restir_pt_initial.h"

namespace serenity::cuda {

namespace {

// The light tiles' block, a thread an entry: the host's choice alone, which
// the kernel, reading only its place, does not assume.
constexpr unsigned tiles_block = 256;

}  // namespace

RestirPtInitialPass::RestirPtInitialPass(const OptixContext& context,
                                         TracedProgram::Records records)
    : kernels_(cuda_modules::restir_pt_initial_kernels),
      power_(kernels_.kernel("restir_pt_light_power")),
      tiles_(kernels_.kernel("restir_pt_light_tiles")),
      tree_(context, optix_modules::restir_pt_initial,
            "__raygen__restir_pt_initial", records),
      mnee_(context, records) {}

void RestirPtInitialPass::record(CUstream stream, CUdeviceptr launch,
                                 const FrameLaunch& frame) const {
  if (frame.path_reservoirs == 0 || frame.primaries == 0 || frame.light == 0 ||
      frame.light_tiles == 0 || frame.light_power == 0 ||
      frame.mnee_requests == 0 || frame.mnee_header == 0) {
    throw CudaError(
        "RestirPtInitialPass: the frame lacks one of its path reservoirs, "
        "primary hits, light buffer, light tiles, power pyramid or MNEE's "
        "work list");
  }
  CUdeviceptr address = launch;
  std::array<void*, 1> arguments{&address};
  cuda::launch(stream, power_,
               LaunchShape{.threads_x = contracts::light_power_threads},
               arguments);
  cuda::launch(stream, tiles_,
               linear_shape(std::uint64_t{contracts::light_tiles} *
                                contracts::light_tile_size,
                            tiles_block),
               arguments);
  tree_.launch(stream, launch, {frame.traced.x, frame.traced.y});
  mnee_.record(stream, launch, frame);
}

}  // namespace serenity::cuda
