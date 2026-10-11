#include "core/output/image_format.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "core/output/pfm.h"
#include "core/output/png.h"

namespace serenity::output {

namespace {

// One row a kind: the list image_format.h describes, read by every lookup
// here, so a kind is named once (ES.3).
struct Kind {
    ImageFormat format;
    std::string_view name;  // on a command line, and the extension
    ImageSource source;
};

constexpr std::array kinds{
    Kind{ImageFormat::png, "png", ImageSource::displayed},
    Kind{ImageFormat::pfm, "pfm", ImageSource::accumulated},
};

// `format`'s row; std::logic_error for a value no kind names (P.6).
const Kind& kind_of(ImageFormat format) {
    for (const Kind& kind : kinds) {
        if (kind.format == format) {
            return kind;
        }
    }
    throw std::logic_error("an image format with no kind in Output's list");
}

[[noreturn]] void refuse_source(ImageFormat format, const std::filesystem::path& path, std::string_view given) {
    throw std::invalid_argument("write_image " + path.string() + ": a " + std::string{kind_of(format).name} +
                                " is not written from " + std::string{given});
}

}  // namespace

std::optional<ImageFormat> image_format_named(std::string_view name) {
    for (const Kind& kind : kinds) {
        if (kind.name == name) {
            return kind.format;
        }
    }
    return std::nullopt;
}

std::string image_format_names() {
    std::string names;
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        names += i == 0 ? "" : (i + 1 == kinds.size() ? " or " : ", ");
        names += kinds[i].name;
    }
    return names;
}

std::string_view extension(ImageFormat format) {
    return kind_of(format).name;
}

ImageSource source(ImageFormat format) {
    return kind_of(format).source;
}

void write_image(ImageFormat format, const std::filesystem::path& path, const DisplayedImage& image) {
    // No default: a kind with no case fails to compile (ES.79); one whose
    // source is not this one is refused before anything is written (I.5).
    switch (format) {
    case ImageFormat::png:
        write_png(path, image.extent, image.rgba);
        return;
    case ImageFormat::pfm:
        refuse_source(format, path, "a displayed frame");
    }
    (void)kind_of(format);  // throws: a value no kind names (P.6)
}

void write_image(ImageFormat format, const std::filesystem::path& path, const contracts::LinearImage& image) {
    switch (format) {
    case ImageFormat::png:
        refuse_source(format, path, "an accumulated image");
    case ImageFormat::pfm:
        write_pfm(path, image);
        return;
    }
    (void)kind_of(format);  // throws: a value no kind names (P.6)
}

}  // namespace serenity::output
