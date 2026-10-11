#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include "core/contracts/linear_image.h"

namespace serenity::output {

// Axis: Output (PFM).
//
// Linear radiance written and read as a Portable Float Map: the format a
// reference is kept in and every image judged against it is written in
// (core/measurement/). pbrt-v4 reads and writes it, and Falcor's error
// measure (ErrorMeasurePass) loads its reference from it, so a reference
// here opens in their tools and in viewers such as tev. Why PFM and not
// OpenEXR: docs/research/2026-10-10-reference.md.
//
// The format, the colour variant only:
//
//   "PF\n"                       three channels a pixel
//   "<width> <height>\n"         decimal, each from 1 to max_image_side
//   "-1.0\n"                     the scale: its sign the floats' byte order,
//                                negative little-endian; its magnitude a
//                                factor on every value
//   width x height x 3 floats    32-bit IEEE 754, rows from the BOTTOM
//
// The rows are stored bottom first, as the format has them, and an image
// here is top first (core/contracts/linear_image.h): write_pfm() and read_pfm()
// turn the rows, and nothing else does. The floats are this machine's own,
// little-endian (a static_assert in pfm.cpp states it, P.5): no byte is
// swapped.
//
// write_pfm(): `image` must hold 3 x width x height floats of an extent
// make_linear_image() accepts, which is checked before anything is written
// (std::invalid_argument, I.5). Each float is written as it is, no scale and
// no clamp, non-finite values included: what a value means is the
// measurement's to judge (error.h refuses them).
//
// write_pfm() creates its file, and never writes over one: the file is
// opened by exclusive create (std::fopen's "x", C11 and C++17's <cstdio>;
// O_CREAT | O_EXCL on this system), which fails if anything is at the path,
// a dangling link included, and is the check itself, made by the file
// system at once, so of two writers racing to one path one creates it and
// the other is refused. That is what lets a reference be "never replaced"
// (measure/options.h), and a frame never land on an earlier run's. An
// existing path is refused by PfmError naming it, left as it was. A write
// that fails after the file is created throws PfmError naming the path and
// removes the file, this call's own. Written through a C stream, not an
// iostream: C++20's file streams have no exclusive create (std::ios::
// noreplace is C++23), a departure from SL.io.3 kept to pfm.cpp.
//
// read_pfm(): every file is untrusted input (SL.io.2), and is refused, by
// PfmError naming the path and what is wrong, before anything is allocated
// for it: a magic other than "PF" (the greyscale "Pf" too); a width or
// height not a decimal integer from 1 to max_image_side (read by
// std::from_chars, E.28, so a sign, a space or a digit past it is
// malformed); a scale other than exactly -1 (a positive one means
// big-endian floats, and a magnitude other than 1 scales every value, as
// pbrt-v4's reader applies it; this program writes neither, and a file
// that has one is refused rather than read as other values than it holds);
// and a length other than the header's plus exactly 3 x width x height x 4
// bytes, truncated or with anything after it. The data's size is computed from
// sides already bounded by max_image_side, so it cannot overflow (ES.103).
// The floats themselves are read as they are.
//
// Not performance-sensitive: one file a reference batch or a measured
// image, some 25 MB at 1920 x 1080, after its frame is done. Read whole
// into the image with one stream operation, its rows then turned in place;
// written a row at a time, bottom first, through the C stream's buffer. No
// copy of the image is made here either way, so a file read holds no more
// than its image (max_image_side's bound). What a caller copies to make
// the image is its own cost (the headless renderer's: headless/options.h,
// docs/research/2026-10-10-reference.md).

class PfmError : public std::runtime_error {
public:
    explicit PfmError(const std::string& what) : std::runtime_error(what) {}
};

void write_pfm(const std::filesystem::path& path, const contracts::LinearImage& image);

contracts::LinearImage read_pfm(const std::filesystem::path& path);

}  // namespace serenity::output
