#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "core/frame/extent.h"

namespace serenity::film {

// Axis: Film (an image of linear radiance, on the CPU).
//
// A rendered image as radiance, before tone mapping: three floats a pixel,
// red, green and blue, rows from the top, with no padding. What a reference
// is kept as and an error is measured on (core/measurement/reference.h,
// error.h), read from and written to Portable Float Maps
// (core/output/pfm.h). The GPU's accumulated image holds a fourth float, a
// pixel's count of samples (metal/film/accumulate.metal.h); it is not kept
// here (from_rgba, below): what an image is judged by is its radiance
// alone.
//
// Plain data with a stated invariant, rgb.size() == 3 x width x height,
// which make_linear_image() establishes and every reader checks before it
// indexes (I.5, SL.con.3); a struct, not a class, since every member is
// public and the invariant is checked where an image arrives, not hidden
// (C.2).

// The largest side an image may have: Metal's largest texture side
// (metal/device/device.h, max_texture_side), which no rendered image
// passes. A file claiming more is refused before anything is allocated for
// it (core/output/pfm.h).
inline constexpr std::size_t max_image_side = 16384;

struct LinearImage {
    frame::Extent extent;
    std::vector<float> rgb;  // 3 x width x height, rows from the top
};

// An image of `extent`, every value 0. Throws std::invalid_argument for a
// side of 0 or past max_image_side.
LinearImage make_linear_image(frame::Extent extent);

// The radiance of the GPU's accumulated image of `extent`, RGBA floats a
// pixel (metal/frame/renderer.h, read_accumulated): each pixel's first three
// floats, its count left out. Throws std::invalid_argument for an extent
// make_linear_image() refuses, or `rgba` not 4 x width x height floats.
LinearImage from_rgba(frame::Extent extent, std::span<const float> rgba);

}  // namespace serenity::film
