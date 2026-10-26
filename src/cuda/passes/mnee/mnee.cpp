#include "cuda/passes/mnee/mnee.h"

#include <array>
#include <cstdint>

#include "core/contracts/mnee_request.h"
#include "cuda/device/error.h"
#include "serenity/cuda_modules/mnee_kernels.h"
#include "serenity/optix_modules/mnee.h"

namespace serenity::cuda {

namespace {

// The walks' kernels' blocks: a thread a list's place.
constexpr unsigned walk_threads = 256;

}  // namespace

namespace {

// The finish program's register limit, by Pipeline::Programs' rule: 168,
// 12 warps an SM, the lowest step's top where OptiX's compile report shows
// no direct spills (152: 20 bytes; the RTX 3090 Ti, fast math,
// 2026-10-21).
constexpr int finish_registers = 168;

}  // namespace

MneePass::MneePass(const OptixContext& context, TracedProgram::Records records)
    : finish_(context, optix_modules::mnee, "__raygen__mnee_finish",
              finish_registers, records),
      kernels_(cuda_modules::mnee_kernels),
      start_(kernels_.kernel("mnee_walk_start")),
      round_(kernels_.kernel("mnee_walk_round")),
      settle_(kernels_.kernel("mnee_settle")) {}

void MneePass::record(CUstream stream, CUdeviceptr launch,
                      const FrameLaunch& frame) const {
  if (frame.mnee_requests == 0 || frame.mnee_header == 0 ||
      frame.mnee_lost == 0 || frame.light == 0 || frame.non_finite == 0 ||
      frame.mnee_walks_a == 0 || frame.mnee_walks_b == 0 ||
      frame.mnee_converged == 0 || frame.mnee_walk_header == 0) {
    throw CudaError(
        "MneePass: the frame lacks its MNEE work list, its walks' lists, its "
        "counter of lost requests or of samples not finite, or the light "
        "buffer it adds to");
  }
  // Each list's capacity, one a pixel (cuda/frame/mnee_list.h).
  const unsigned capacity = frame.frame.width * frame.frame.height;
  const LaunchShape lists{
      .blocks_x = (capacity + walk_threads - 1) / walk_threads,
      .threads_x = walk_threads};
  CUdeviceptr address = launch;
  std::array<void*, 1> arguments{&address};
  cuda::launch(stream, start_, lists, arguments);
  for (unsigned round = 1; round < contracts::mnee_rounds; ++round) {
    // Round `round` reads the list the round before wrote and writes the
    // other, its count set to 0 first (contract 16,
    // core/contracts/mnee_request.h).
    unsigned from = (round - 1) % contracts::mnee_walk_lists;
    const unsigned into = round % contracts::mnee_walk_lists;
    check(
        cuMemsetD32Async(frame.mnee_walk_header + into * sizeof(std::uint32_t),
                         0, 1, stream),
        "MneePass: cuMemsetD32Async");
    std::array<void*, 2> round_arguments{&address, &from};
    cuda::launch(stream, round_, lists, round_arguments);
  }
  finish_.launch(stream, launch, {capacity, 1});
  cuda::launch(stream, settle_, LaunchShape{}, arguments);
}

}  // namespace serenity::cuda
