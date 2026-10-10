// PNG output: a file that decodes to exactly the pixels given, rows from the
// top, R, G, B, A; and a wrong-sized buffer refused before anything is
// written.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include <doctest/doctest.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

#include "core/output/png.h"

using serenity::frame::Extent;
using serenity::output::PngError;
using serenity::output::write_png;

namespace {

// A file of the test's own in the system's temporary directory, named
// by the process so two runs at once (both presets) never share one,
// and removed however the test ends (R.1).
class TemporaryFile {
public:
    explicit TemporaryFile(const std::string& stem)
        : path_(std::filesystem::temp_directory_path() /
                ("serenity-" + stem + "-" + std::to_string(::getpid()) + ".png")) {}
    ~TemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    TemporaryFile(TemporaryFile&&) = delete;
    TemporaryFile& operator=(TemporaryFile&&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

std::uint32_t big_endian(const std::vector<unsigned char>& bytes, std::size_t at) {
    return (std::uint32_t{bytes[at]} << 24) | (std::uint32_t{bytes[at + 1]} << 16) |
           (std::uint32_t{bytes[at + 2]} << 8) | std::uint32_t{bytes[at + 3]};
}

}  // namespace

TEST_CASE("a PNG holds exactly the pixels given: its size in the header, each channel in its place") {
    // A different value in every channel of every pixel: a swapped channel,
    // a row written bottom up, or a wrong stride shows. Not square, so
    // width and height cannot be swapped unseen.
    constexpr Extent size{5, 3};
    std::vector<std::uint8_t> rgba(std::size_t{size.width} * size.height * 4);
    for (std::size_t i = 0; i < rgba.size(); ++i) {
        rgba[i] = static_cast<std::uint8_t>(7 + 3 * i);
    }
    const TemporaryFile file("png-test");
    write_png(file.path(), size, rgba);

    std::ifstream in(file.path(), std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), {});
    REQUIRE(bytes.size() > 33);
    constexpr std::array<unsigned char, 8> signature{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    CHECK(std::equal(signature.begin(), signature.end(), bytes.begin()));
    CHECK(bytes[12] == 'I');
    CHECK(bytes[15] == 'R');
    CHECK(big_endian(bytes, 16) == size.width);
    CHECK(big_endian(bytes, 20) == size.height);

    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* decoded =
        stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &width, &height, &channels, 0);
    REQUIRE(decoded != nullptr);
    const std::vector<std::uint8_t> pixels(decoded, decoded + rgba.size());
    stbi_image_free(decoded);
    CHECK(width == static_cast<int>(size.width));
    CHECK(height == static_cast<int>(size.height));
    CHECK(channels == 4);
    CHECK(pixels == rgba);
}

TEST_CASE("a buffer of the wrong size is refused, and nothing is written") {
    const TemporaryFile file("png-wrong");
    const std::vector<std::uint8_t> rgba(3 * 2 * 4 - 1, 0);
    CHECK_THROWS_AS(write_png(file.path(), Extent{3, 2}, rgba), PngError);
    CHECK_THROWS_AS(write_png(file.path(), Extent{0, 2}, {}), PngError);
    CHECK_FALSE(std::filesystem::exists(file.path()));
}

TEST_CASE("a path that cannot be written is an error") {
    const std::vector<std::uint8_t> rgba(4, 0);
    CHECK_THROWS_AS(write_png("/no/such/directory/x.png", Extent{1, 1}, rgba), PngError);
}
