#include "cuda/passes/path/path.h"

#include "serenity/optix_modules/path.h"

namespace serenity::cuda {

namespace {

// The path tracer's register limit, by Pipeline::Programs' rule: 128, 16
// warps an SM, where OptiX's compile report shows no direct spills (the
// RTX 3090 Ti, fast math, 2026-10-21).
constexpr int path_registers = 128;

}  // namespace

PathPass::PathPass(const OptixContext& context, TracedProgram::Records records)
    : program_(context, optix_modules::path, "__raygen__path", path_registers,
               records) {}

void PathPass::record(CUstream stream, CUdeviceptr launch,
                      frame::Extent size) const {
  program_.launch(stream, launch, size);
}

}  // namespace serenity::cuda
