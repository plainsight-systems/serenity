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

LightGlows::LightGlows(const Device& device, Submission& submission, std::uint32_t lights, bool glows)
    : lights_(lights), array_(device, submission, std::as_bytes(std::span<const float>(ones(lights))),
                              glows ? frames_in_flight : 1u) {}

std::span<float> LightGlows::glows(std::uint32_t slot) const {
    const std::span<std::byte> bytes = array_.bytes(slot);
    // The bytes were made from floats (the constructor), in a buffer whose
    // copies start 256-byte aligned: they are floats.
    return {reinterpret_cast<float*>(bytes.data()), lights_};
}

}  // namespace serenity::metal
