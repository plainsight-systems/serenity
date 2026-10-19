#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/frame/extent.h"
#include "cuda/device/buffer.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: GPU backend.
//
// An image a frame renders into when there is no window, as Metal's
// offscreen target is (metal/device/offscreen.h): RGBA, 8 bits a channel,
// rows from the top, tightly packed (width x 4 bytes a row), in device
// memory made once at the run's size (MEM.9). It belongs to the headless
// renderer and the tests, never to a frame loop: read_rgba() waits for
// every committed frame and copies the image to the host (GPU.1: a
// readback, the point of the program that asks for it).
//
// Throws CudaError if the size is empty or the memory cannot be made.
class Offscreen {
 public:
  Offscreen(Submission& submission, frame::Extent size);

  Offscreen(const Offscreen&) = delete;
  Offscreen& operator=(const Offscreen&) = delete;
  Offscreen(Offscreen&&) = delete;
  Offscreen& operator=(Offscreen&&) = delete;
  ~Offscreen() = default;

  frame::Extent size() const noexcept { return size_; }
  // The image's device address, which a pass writes.
  CUdeviceptr address() const noexcept { return image_.address(); }

  // The bytes read_rgba() fills: width x height x 4.
  std::size_t rgba_size() const noexcept;

  // Waits for every committed frame (Submission::finish) and copies the
  // image into `out`, exactly rgba_size() bytes; throws CudaError
  // otherwise, or for a fault in a frame waited for.
  void read_rgba(std::span<std::uint8_t> out) const;

 private:
  Submission& submission_;
  frame::Extent size_;
  DeviceBuffer image_;
};

}  // namespace serenity::cuda
