#pragma once

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "core/frame/extent.h"

namespace serenity::cuda {

// Axis: GPU backend.
//
// A module of kernels: one pass's `.cu` compiled by nvcc at build time to a
// fatbin for sm_86 and embedded in the program (cmake/CudaModules.cmake),
// loaded from its bytes at construction, as Metal's library is
// (metal/device/library.h) and as Cycles loads its kernels: what a kernel
// computes is fixed by the source revision, never by where the build put a
// file. Unloaded with this object (R.1), which a pass owns, released after
// the frames in flight (submission.h, Lifetime).
//
// Throws CudaError if the driver refuses the bytes, or names no kernel.
class Module {
 public:
  explicit Module(std::span<const std::byte> fatbin);

  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Module(Module&&) = delete;
  Module& operator=(Module&&) = delete;
  ~Module();

  // The kernel `name`, its extern "C" name; throws CudaError for none.
  CUfunction kernel(std::string_view name) const;

 private:
  CUmodule module_ = nullptr;
};

// The threads a block of a per-pixel launch: 16 x 16, a common shape for
// image kernels on NVIDIA's GPUs (eight warps), tuned per pass when a pass
// is (Phase 6's tuning, not the port's).
inline constexpr unsigned per_pixel_block = 16;

// A launch's blocks and the threads in each, in two dimensions (I.24:
// named, so the two pairs cannot pass in each other's places).
struct LaunchShape {
  unsigned blocks_x = 1;
  unsigned blocks_y = 1;
  unsigned threads_x = 1;
  unsigned threads_y = 1;
};

// Launches `kernel` on `stream` in `shape`, with `arguments` as
// launch_per_pixel() takes them. Throws CudaError if the launch fails.
void launch(CUstream stream, CUfunction kernel, LaunchShape shape,
            std::span<void*> arguments);

// The shape of `count` threads in one dimension, in blocks of `block`, the
// last block cut short by the kernel's own guard on its count.
LaunchShape linear_shape(std::uint64_t count, unsigned block);

// Launches `kernel` on `stream` over one thread a pixel of `grid`, in blocks
// of per_pixel_block squared, with `arguments`, each a pointer to one of
// the kernel's parameters in order, which the driver copies at the launch
// (so they may be locals). Kernels return past the grid's edge. Throws
// CudaError if the launch fails; a fault while it runs is reported by the
// submission's next wait (error.h).
void launch_per_pixel(CUstream stream, CUfunction kernel, frame::Extent grid,
                      std::span<void*> arguments);

}  // namespace serenity::cuda
