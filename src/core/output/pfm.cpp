#include "core/output/pfm.h"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

namespace serenity::output {

namespace {

// The floats are written and read as this machine holds them, no byte
// swapped: what "-1.0", little-endian, promises of them (pfm.h, P.5).
static_assert(std::endian::native == std::endian::little, "a PFM's floats are written as little-endian");
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
              "a PFM's floats are 32-bit IEEE 754");

constexpr std::size_t channels = 3;  // red, green, blue: the colour variant, "PF"
constexpr std::string_view magic = "PF";
constexpr std::string_view scale_written = "-1.0";  // little-endian, every value as it is
// The longest header line read before it is refused: "16384 16384" is 11
// bytes, so a line past this is no header of this format, and a binary file
// is not scanned to its end for a newline (SL.io.2).
constexpr std::size_t longest_header_line = 32;

// The bytes an image's floats are, for the stream's char interface: char
// may alias any object ([basic.lval]), and the floats are this machine's
// own (above). Reading into them writes the object representation of
// floats that already exist (LIFE.4). The only casts in this file, kept to
// these two (I.30).
std::span<const char> bytes_of(std::span<const float> floats) {
    return {reinterpret_cast<const char*>(floats.data()), floats.size_bytes()};
}
std::span<char> writable_bytes_of(std::span<float> floats) {
    return {reinterpret_cast<char*>(floats.data()), floats.size_bytes()};
}

// A byte count as the stream's: every count here is at most a 16384 x 16384
// image's 3 GiB, far inside a 64-bit streamsize (ES.46).
std::streamsize stream_size(std::size_t bytes) {
    static_assert(std::numeric_limits<std::streamsize>::max() >= 0x7fff'ffff'ffffLL,
                  "the stream counts the bytes of an image of max_image_side");
    return static_cast<std::streamsize>(bytes);
}

[[noreturn]] void refuse(const std::filesystem::path& path, const std::string& why) {
    throw PfmError("cannot read " + path.string() + " as a PFM: " + why);
}

// One line of the header, its newline consumed and not kept; refused if the
// file ends first or it runs past longest_header_line. Read a character at a
// time, where it has to be (SL.io.1): std::getline would read a binary
// file's first "line" whole, whatever its length, before any bound applied.
std::string header_line(std::istream& in, const std::filesystem::path& path, std::string_view which) {
    std::string line;
    char c = 0;
    while (in.get(c)) {
        if (c == '\n') {
            return line;
        }
        if (line.size() == longest_header_line) {
            refuse(path, "its " + std::string{which} + " line is longer than " +
                             std::to_string(longest_header_line) + " bytes");
        }
        line.push_back(c);
    }
    refuse(path, "it ends in its header, before the newline of its " + std::string{which} + " line");
}

// The whole of `text` as a side from 1 to max_image_side, or none:
// std::from_chars reads no locale and sets no errno (E.28), and takes no
// sign or space.
std::optional<std::uint32_t> side(std::string_view text) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0 || value > contracts::max_image_side) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value);  // at most max_image_side (ES.46)
}

// The scale line, exactly -1, else refused: a positive scale is big-endian
// floats, and a magnitude other than 1 a factor on every value (pfm.h).
void check_scale(std::string_view text, const std::filesystem::path& path) {
    double value = 0.0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        refuse(path, "its scale '" + std::string{text} + "' is not a number");
    }
    if (value != -1.0) {  // a NaN is not -1 either
        refuse(path, "its scale is " + std::string{text} +
                         ", not -1: only little-endian floats, each its value as it is, are read");
    }
}

// What a header says: the image's extent, and how many bytes it took.
struct Header {
    frame::Extent extent;
    std::size_t bytes = 0;
};

// The three lines of the header, each bounded, read and refused before
// anything is allocated for the data (SL.io.2).
Header read_header(std::istream& in, const std::filesystem::path& path) {
    const std::string first = header_line(in, path, "magic");
    if (first != magic) {
        refuse(path, "its magic is '" + first + "', not 'PF' (three channels a pixel)");
    }
    const std::string sides = header_line(in, path, "size");
    const std::size_t space = sides.find(' ');
    const std::optional<std::uint32_t> width =
        space == std::string::npos ? std::nullopt : side(std::string_view{sides}.substr(0, space));
    const std::optional<std::uint32_t> height =
        space == std::string::npos ? std::nullopt : side(std::string_view{sides}.substr(space + 1));
    if (!width || !height) {
        refuse(path, "its size '" + sides + "' is not WIDTH HEIGHT, each a whole number from 1 to " +
                         std::to_string(contracts::max_image_side));
    }
    const std::string scale = header_line(in, path, "scale");
    check_scale(scale, path);
    constexpr std::size_t newlines = 3;  // one a line, read and not kept
    return Header{.extent = {*width, *height}, .bytes = first.size() + sides.size() + scale.size() + newlines};
}

}  // namespace

void write_pfm(const std::filesystem::path& path, const contracts::LinearImage& image) {
    // Contract 13's check, before the file is opened (I.5).
    (void)contracts::checked_values(image, "write_pfm " + path.string());
    const frame::Extent extent = image.extent;
    // Sides bounded by the check, so this cannot overflow (ES.103).
    const std::size_t row_floats = std::size_t{extent.width} * channels;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw PfmError("cannot write " + path.string() + ": it cannot be opened");
    }
    // std::to_string, not the stream's operator<<: no locale's grouping can
    // reach the header (environmental determinism).
    const std::string header = std::string{magic} + "\n" + std::to_string(extent.width) + " " +
                               std::to_string(extent.height) + "\n" + std::string{scale_written} + "\n";
    out.write(header.data(), stream_size(header.size()));
    // Rows from the bottom, as the format has them (pfm.h); each written
    // from the image as it is, through the stream's buffer, with no copy.
    const std::span<const float> rgb = image.rgb;
    for (std::size_t r = 0; r < extent.height; ++r) {
        const std::size_t row = extent.height - 1 - r;
        const std::span<const char> bytes = bytes_of(rgb.subspan(row * row_floats, row_floats));
        out.write(bytes.data(), stream_size(bytes.size()));
    }
    out.close();
    if (!out) {
        throw PfmError("cannot write " + path.string() + ": the write failed");
    }
}

contracts::LinearImage read_pfm(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        refuse(path, "it cannot be opened");
    }
    std::error_code error;
    const std::uintmax_t file_bytes = std::filesystem::file_size(path, error);
    if (error) {
        refuse(path, "its size cannot be read: " + error.message());
    }

    // The data's size, from sides already bounded, cannot overflow
    // (ES.103); the file's is checked against it before the image is made.
    const Header header = read_header(in, path);
    const frame::Extent extent = header.extent;
    const std::size_t data_bytes = std::size_t{extent.width} * extent.height * channels * sizeof(float);
    const std::uintmax_t expected = std::uintmax_t{header.bytes} + data_bytes;
    if (file_bytes < expected) {
        refuse(path, "it is truncated: " + std::to_string(file_bytes) + " bytes, where its header says " +
                         std::to_string(expected));
    }
    if (file_bytes > expected) {
        refuse(path, std::to_string(file_bytes - expected) + " bytes follow its " + std::to_string(extent.width) +
                         " x " + std::to_string(extent.height) + " floats");
    }

    contracts::LinearImage image = contracts::make_linear_image(extent);
    const std::span<char> bytes = writable_bytes_of(image.rgb);
    in.read(bytes.data(), stream_size(bytes.size()));
    // Checked again as read, in case the file changed since its size was
    // taken.
    if (in.gcount() != stream_size(bytes.size())) {
        refuse(path, "it is truncated: it ended within its floats");
    }
    if (in.peek() != std::ifstream::traits_type::eof()) {
        refuse(path, "bytes follow its floats");
    }

    // The file's rows from the bottom; the image's from the top (pfm.h).
    const std::size_t row_floats = std::size_t{extent.width} * channels;
    const std::span<float> rgb = image.rgb;
    for (std::size_t top = 0, bottom = extent.height - 1u; top < bottom; ++top, --bottom) {
        std::ranges::swap_ranges(rgb.subspan(top * row_floats, row_floats),
                                 rgb.subspan(bottom * row_floats, row_floats));
    }
    return image;
}

}  // namespace serenity::output
