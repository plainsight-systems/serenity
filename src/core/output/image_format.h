#pragma once

#include <optional>
#include <string_view>

namespace serenity::output {

// Axis: Output (which writer).
//
// The kinds of file a rendered frame is written as, and their names on a
// command line: the Output family's list, so a new writer (OpenEXR, say)
// adds a kind here and its file beside png.h and pfm.h, and no program
// that writes frames enumerates the kinds itself (headless/options.h
// stores one).
//
//   png  the frame as displayed, 8 bits a channel, tone mapped (png.h)
//   pfm  linear radiance in floats, before tone mapping (pfm.h)

enum class ImageFormat {
    png,
    pfm,
};

// The kind `name` names, exactly ("png", "pfm"), or none.
std::optional<ImageFormat> image_format_named(std::string_view name);

// Its file name's extension, without the dot: what it is named for.
std::string_view extension(ImageFormat format);

}  // namespace serenity::output
