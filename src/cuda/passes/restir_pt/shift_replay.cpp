#include "cuda/passes/restir_pt/shift_replay.h"

#include <array>
#include <cstdint>

#include "core/contracts/path_reuse_settings.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

ShiftReplay::ShiftReplay(const OptixContext& context,
                         std::span<const std::byte> programs,
                         std::span<const std::byte> kernels, Names names,
                         TracedProgram::Records records)
    : shift_(context, programs, names.shift, records),
      replay_(context, programs, names.replay, records),
      kernels_(kernels),
      merge_(kernels_.kernel(names.merge)) {}

void ShiftReplay::record(CUstream stream, CUdeviceptr launch,
                         const FrameLaunch& frame) const {
  if (frame.shifts == 0 || frame.replays == 0 || frame.replay_counter == 0) {
    throw CudaError(
        "ShiftReplay: the frame lacks the shifted paths or the replay list");
  }
  const frame::Extent traced{frame.traced.x, frame.traced.y};
  shift_.launch(stream, launch, traced);
  // The list's capacity: max_path_neighbors places a pixel of the arrays'
  // grid (core/passes/restir_pt_arrays.h).
  replay_.launch(
      stream, launch,
      {frame.frame.width * frame.frame.height * contracts::max_path_neighbors,
       1});
  check(cuMemsetD32Async(frame.replay_counter, 0, 1, stream),
        "ShiftReplay: cuMemsetD32Async");
  CUdeviceptr address = launch;
  std::array<void*, 1> arguments{&address};
  launch_per_pixel(stream, merge_, traced, arguments);
}

}  // namespace serenity::cuda
