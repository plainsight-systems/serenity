#pragma once

// Contract 13: the linear image. Owned by Film; read by Output and
// Measurement.
//
// A rendered image as radiance, before tone mapping, on the CPU: three
// floats a pixel, red, green and blue, rows from the top, with no padding.
// What a reference is kept as and an error is measured on
// (core/measurement/reference.h, error.h), and what Portable Float Maps
// hold (core/output/pfm.h): the value those families pass, so neither
// depends on the other or on Film (file-mapping.md: families depend on
// contracts). The GPU's accumulated image holds a fourth float, a pixel's
// count of samples (metal/film/accumulate.metal.h); it is not kept here
// (from_rgba, below): what an image is judged by is its radiance alone.
//
// Plain data with a stated invariant, rgb.size() == 3 x width x height,
// which make_linear_image() establishes and every reader checks before it
// indexes (I.5, SL.con.3); a struct, not a class, since every member is
// public and the invariant is checked where an image arrives, not hidden
// (C.2). The two functions are declared here and defined by Film, the
// owner (core/film/linear_image.cpp).
//
// max_image_side bounds what a file may make a reader allocate: a side of
// 16384 is at most 16384 x 16384 x 12 bytes, 3 GiB, for an image read
// whole, and past any image this program is asked for (the display is 3456
// x 2234). It is the core's own bound, for the CPU's images; a backend's
// texture limit is that backend's (metal/device/device.h), and happens to
// be the same number today.

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#include "core/frame/extent.h"

namespace serenity::contracts {

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

}  // namespace serenity::contracts
