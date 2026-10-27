#pragma once

#include <optional>
#include <span>

#include "core/frame/extent.h"

namespace serenity::metal {

// Axis: Pass (the denoiser's network on the Neural Engine).
//
// The grid a light runs at when its network is compiled for a fixed set of
// shapes (neural_engine.h, Metal's alone): the smallest of `rungs` that
// holds the light's padded grid (core/denoise/unet.h, unet_padded()) on
// both axes, smallest by pixels, the narrower of two the same; the light
// is padded to it with zeros, as to the U-Net's alignment. None if no rung
// holds it: a light past the largest is refused by the denoise pass, not
// cropped. Pure (F.8), tested apart from Metal.
std::optional<frame::Extent> smallest_rung(std::span<const frame::Extent> rungs,
                                           frame::Extent light) noexcept;

}  // namespace serenity::metal
