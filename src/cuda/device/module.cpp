#include "cuda/device/module.h"

#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

Module::Module(std::span<const std::byte> fatbin) {
  if (fatbin.empty()) {
    throw CudaError("Module: no bytes to load");
  }
  check(cuModuleLoadData(&module_, fatbin.data()), "cuModuleLoadData");
}

Module::~Module() { (void)cuModuleUnload(module_); }

CUfunction Module::kernel(std::string_view name) const {
  CUfunction function = nullptr;
  check(cuModuleGetFunction(&function, module_, std::string(name).c_str()),
        "cuModuleGetFunction " + std::string(name));
  return function;
}

void launch_per_pixel(CUstream stream, CUfunction kernel, frame::Extent grid,
                      std::span<void*> arguments) {
  const unsigned blocks_x =
      (grid.width + per_pixel_block - 1) / per_pixel_block;
  const unsigned blocks_y =
      (grid.height + per_pixel_block - 1) / per_pixel_block;
  check(
      cuLaunchKernel(kernel, blocks_x, blocks_y, 1, per_pixel_block,
                     per_pixel_block, 1, 0, stream, arguments.data(), nullptr),
      "cuLaunchKernel");
}

}  // namespace serenity::cuda
