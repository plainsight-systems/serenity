#include "cuda/frame/traced_program.h"

#include "cuda/frame/frame_launch.h"

namespace serenity::cuda {

namespace {

// The backend's hit groups' intersection programs, in their groups' order
// (structure_block.h): none for the triangles, whose hits are OptiX's, and
// the seed's.
constexpr std::array<std::string_view, 2> intersections{"",
                                                        "__intersection__seed"};
static_assert(triangles_group == 0 && seed_group == 1);

}  // namespace

TracedProgram::TracedProgram(const OptixContext& context,
                             std::span<const std::byte> module,
                             std::string_view ray_generation, Records records)
    : pipeline_(context, module,
                Pipeline::Programs{
                    .ray_generation = ray_generation,
                    .intersections = intersections,
                    .launch_parameters = "launch",
                    .payload_values = trace_payload_values,
                    .hit_records = records,
                }) {}

void TracedProgram::launch(CUstream stream, CUdeviceptr launch,
                           frame::Extent grid) const {
  pipeline_.launch(stream, launch, sizeof(FrameLaunch), grid);
}

}  // namespace serenity::cuda
