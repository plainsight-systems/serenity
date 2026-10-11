#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "core/contracts/linear_image.h"
#include "core/frame/extent.h"

namespace serenity::output {

// Axis: Output (which writer).
//
// The kinds of file a rendered frame is written as: the Output family's
// list, and the one place a kind is named. For each kind, its name on a
// command line, its file name's extension, and its source, what of a frame
// its writer takes:
//
//   kind  source       written as
//   png   displayed    the frame as displayed, 8 bits a channel, tone
//                      mapped: for looking at (png.h)
//   pfm   accumulated  the graph's accumulated image, linear radiance in
//                      floats before tone mapping, its pixels' counts left
//                      out: for measuring (pfm.h, contract 13)
//
// A program that writes frames stores a kind (headless/options.h), reads
// back what its source() names, and hands it to write_image(); it names no
// kind and branches on none. Adding a kind (OpenEXR, say) adds a row to
// the table in image_format.cpp, an enumerator here, its writer beside
// png.h and pfm.h, and its case in write_image(): no program changes,
// unless the kind's source is a new one.
//
// Not performance-sensitive: a lookup in a table of two rows, once a run;
// write_image() is its writer's cost.

enum class ImageFormat {
    png,
    pfm,
};

// What a frame is read back as for a kind's writer: a source, which a
// backend knows how to read, where a kind is Output's.
enum class ImageSource {
    displayed,    // the frame as shown, 8-bit RGBA (metal/device/offscreen.h)
    accumulated,  // the accumulated image's radiance (contract 13, from_rgba)
};

// The kind a program writes when none is asked for.
inline constexpr ImageFormat default_format = ImageFormat::png;

// The kind `name` names, exactly ("png", "pfm"), or none.
std::optional<ImageFormat> image_format_named(std::string_view name);

// Every name image_format_named() accepts, for a diagnostic: "png or pfm".
std::string image_format_names();

// Its file name's extension, without the dot: what it is named for.
std::string_view extension(ImageFormat format);

// What its writer takes.
ImageSource source(ImageFormat format);

// A frame as displayed: `rgba`, 4 bytes a pixel, rows from the top, of
// `extent` (png.h's contract). A view, held for the call it is made for.
struct DisplayedImage {
    frame::Extent extent;
    std::span<const std::uint8_t> rgba;
};

// Writes `image` to `path` as `format`, by that kind's writer, with its
// contract and its errors (png.h, PngError; pfm.h, PfmError). A kind
// whose source is not the one given is refused by std::invalid_argument
// before anything is written (I.5): an accumulated image is not written as
// a PNG, nor a displayed frame as a PFM. extension() and source() throw
// std::logic_error for a value no kind names (P.6), as write_image() does.
void write_image(ImageFormat format, const std::filesystem::path& path, const DisplayedImage& image);
void write_image(ImageFormat format, const std::filesystem::path& path, const contracts::LinearImage& image);

}  // namespace serenity::output
