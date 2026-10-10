#include "metal/scene/light_glows.h"

#include <algorithm>
#include <vector>

namespace serenity::metal {

namespace {

// `lights` factors of 1, at least one: a buffer must have bytes.
std::vector<float> ones(std::uint32_t lights) {
    return std::vector<float>(std::max<std::uint32_t>(lights, 1u), 1.0f);
}

}  // namespace

LightGlows::LightGlows(const Device& device, Submission& submission, std::uint32_t lights, bool glowing)
    : lights_(lights), array_(device, submission, std::as_bytes(std::span<const float>(ones(lights))),
                              glowing ? FrameArray::Copies::per_frame : FrameArray::Copies::one) {}

std::span<float> LightGlows::glows(std::uint32_t slot) {
    // The bytes were made from floats (the constructor), at least one: a
    // scene with no sphere lights has one, unread, and is given none.
    return array_.view<float>(slot).first(lights_);
}

}  // namespace serenity::metal
