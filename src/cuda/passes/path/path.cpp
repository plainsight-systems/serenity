#include "cuda/passes/path/path.h"

#include <cstring>

#include "serenity/optix_modules/path.h"

namespace serenity::cuda {

namespace {

// The parameters a launch starts from, before the first frame writes its
// own: zeroes, unread.
constexpr PathLaunch no_launch{};

}  // namespace

PathPass::PathPass(const OptixContext& context, Submission& submission,
                   std::span<const Pipeline::HitData> structures)
    : pipeline_(context, optix_modules::path,
                Pipeline::Programs{
                    .ray_generation = "__raygen__path",
                    .intersection = "__intersection__shape",
                    .launch_parameters = "launch",
                    // the shape a shadow ray ignores (trace.cuh)
                    .payload_values = 1,
                    .hit_records = structures,
                }),
      launch_(submission, std::as_bytes(std::span(&no_launch, 1))) {}

void PathPass::record(const FrameSlot& frame, const PathLaunch& launch) {
  std::memcpy(launch_.bytes(frame.slot).data(), &launch, sizeof(launch));
  launch_.upload(frame);
  pipeline_.launch(frame.stream, launch_.address(), sizeof(launch),
                   {launch.frame.width, launch.frame.height});
}

}  // namespace serenity::cuda
