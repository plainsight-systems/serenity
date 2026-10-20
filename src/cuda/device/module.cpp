#include "cuda/device/module.h"

#include <algorithm>
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

void launch(CUstream stream, CUfunction kernel, LaunchShape shape,
            std::span<void*> arguments) {
  check(
      cuLaunchKernel(kernel, shape.blocks_x, shape.blocks_y, 1, shape.threads_x,
                     shape.threads_y, 1, 0, stream, arguments.data(), nullptr),
      "cuLaunchKernel");
}

LaunchShape linear_shape(std::uint64_t count, unsigned block) {
  return LaunchShape{
      .blocks_x = static_cast<unsigned>(
          std::max<std::uint64_t>((count + block - 1) / block, 1)),
      .threads_x = block,
  };
}

void launch_per_pixel(CUstream stream, CUfunction kernel, frame::Extent grid,
                      std::span<void*> arguments) {
  launch(stream, kernel,
         LaunchShape{
             .blocks_x = (grid.width + per_pixel_block - 1) / per_pixel_block,
             .blocks_y = (grid.height + per_pixel_block - 1) / per_pixel_block,
             .threads_x = per_pixel_block,
             .threads_y = per_pixel_block,
         },
         arguments);
}

}  // namespace serenity::cuda
