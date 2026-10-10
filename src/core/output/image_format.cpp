#include "core/output/image_format.h"

#include <stdexcept>

namespace serenity::output {

std::optional<ImageFormat> image_format_named(std::string_view name) {
    if (name == "png") {
        return ImageFormat::png;
    }
    if (name == "pfm") {
        return ImageFormat::pfm;
    }
    return std::nullopt;
}

std::string_view extension(ImageFormat format) {
    // No default: a kind without an extension fails to compile (ES.79).
    switch (format) {
    case ImageFormat::png:
        return "png";
    case ImageFormat::pfm:
        return "pfm";
    }
    // Reached only by a value no enumerator names: not a file to name (P.6).
    throw std::logic_error("extension: an image format with no extension");
}

}  // namespace serenity::output
