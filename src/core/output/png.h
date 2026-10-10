#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

#include "core/frame/extent.h"

namespace serenity::output {

// Axis: Output (PNG).
//
// Writes an 8-bit RGBA image as a PNG: the format for sharing a frame. The
// encoding is stb_image_write's, pinned by commit (NOTICE); this file is the
// only one that includes it.
//
// Contract:
//   - `rgba` holds extent.width x extent.height pixels, 4 bytes each in the
//     order R, G, B, A, rows from the top, with no padding between rows. Any
//     other length is a PngError, checked before anything is written.
//   - The values are written as they are: no gamma, no tone mapping. What
//     they mean is decided before they get here.
//   - Failure throws PngError naming the path (E.2, E.14). A partial file may be
//     left behind; nothing reads a file whose write failed.
//
// Not performance-sensitive: used by the headless renderer, which is not
// held to the frame budget. Compressing a frame at the display's resolution
// takes tens of milliseconds, on the CPU, after the frame is done.

class PngError : public std::runtime_error {
public:
    explicit PngError(const std::string& what) : std::runtime_error(what) {}
};

void write_png(const std::filesystem::path& path, frame::Extent extent,
               std::span<const std::uint8_t> rgba);

}  // namespace serenity::output
