// The test pattern (test_pattern.h): red across, green down, blue pulsing
// once every four seconds, from the frame's constants alone, as Metal's
// (metal/passes/test_pattern/test_pattern.metal), each channel written as
// an 8-bit unorm, rounded to nearest.

#include <cstdint>

#include "core/contracts/frame_constants.h"

namespace {

constexpr float pulse_seconds = 4.0f;  // blue's period
constexpr float pi = 3.14159265358979323846f;

__device__ std::uint8_t unorm8(float v) {
  return static_cast<std::uint8_t>(
      __float2uint_rn(fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f));
}

}  // namespace

extern "C" __global__ void test_pattern(
    serenity::contracts::FrameConstants frame, uchar4* target) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= frame.width || y >= frame.height) {
    return;
  }
  const float red =
      frame.width > 1u ? float(x) / float(frame.width - 1u) : 0.0f;
  const float green =
      frame.height > 1u ? float(y) / float(frame.height - 1u) : 0.0f;
  const float blue =
      0.5f + 0.5f * sinf(2.0f * pi * frame.time_seconds / pulse_seconds);
  target[y * frame.width + x] =
      make_uchar4(unorm8(red), unorm8(green), unorm8(blue), 255);
}
