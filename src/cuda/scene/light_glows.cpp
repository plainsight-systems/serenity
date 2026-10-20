#include "cuda/scene/light_glows.h"

#include <algorithm>
#include <vector>

namespace serenity::cuda {

namespace {

// `lights` factors of 1, at least one: an array must have bytes.
std::vector<float> ones(std::uint32_t lights) {
  return std::vector<float>(std::max<std::uint32_t>(lights, 1u), 1.0f);
}

}  // namespace

LightGlows::LightGlows(Submission& submission, std::uint32_t lights)
    : lights_(lights),
      array_(submission, std::as_bytes(std::span<const float>(ones(lights)))) {}

}  // namespace serenity::cuda
