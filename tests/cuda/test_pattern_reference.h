#pragma once

// The test pattern as the CPU computes it (cuda/passes/test_pattern/
// test_pattern.cu says what it draws), for the tests to compare the GPU's
// with: each channel an 8-bit unorm, rounded to nearest, ties to even.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace serenity::tests {

inline std::uint8_t unorm8(float v) {
  return static_cast<std::uint8_t>(
      std::nearbyint(std::clamp(v, 0.0f, 1.0f) * 255.0f));
}

// Pixel (x, y) of a `width` x `height` pattern at `seconds`: R, G, B, A.
inline std::array<std::uint8_t, 4> test_pattern_pixel(unsigned x, unsigned y,
                                                      unsigned width,
                                                      unsigned height,
                                                      float seconds) {
  constexpr float pi = 3.14159265358979323846f;
  const float red = width > 1u
                        ? static_cast<float>(x) / static_cast<float>(width - 1u)
                        : 0.0f;
  const float green =
      height > 1u ? static_cast<float>(y) / static_cast<float>(height - 1u)
                  : 0.0f;
  const float blue = 0.5f + 0.5f * std::sin(2.0f * pi * seconds / 4.0f);
  return {unorm8(red), unorm8(green), unorm8(blue), 255};
}

}  // namespace serenity::tests
