// PNG output: a valid file of the right size, and a wrong-sized buffer
// refused before anything is written.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include <doctest/doctest.h>

#include "core/output/png.h"

using serenity::frame::Extent;
using serenity::output::PngError;
using serenity::output::write_png;

namespace {

std::filesystem::path temp_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

std::uint32_t big_endian(const std::vector<unsigned char>& bytes, std::size_t at) {
    return (std::uint32_t{bytes[at]} << 24) | (std::uint32_t{bytes[at + 1]} << 16) |
           (std::uint32_t{bytes[at + 2]} << 8) | std::uint32_t{bytes[at + 3]};
}

}  // namespace

TEST_CASE("a PNG is written with its signature and its size in the header") {
    const auto path = temp_path("serenity-png-test.png");
    const std::vector<std::uint8_t> rgba(3 * 2 * 4, 0x80);
    write_png(path, Extent{3, 2}, rgba);

    std::ifstream file(path, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), {});
    std::filesystem::remove(path);

    REQUIRE(bytes.size() > 33);
    const unsigned char signature[] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(bytes[i] == signature[i]);
    }
    CHECK(bytes[12] == 'I');
    CHECK(bytes[15] == 'R');
    CHECK(big_endian(bytes, 16) == 3);  // width
    CHECK(big_endian(bytes, 20) == 2);  // height
}

TEST_CASE("a buffer of the wrong size is refused, and nothing is written") {
    const auto path = temp_path("serenity-png-wrong.png");
    std::filesystem::remove(path);
    const std::vector<std::uint8_t> rgba(3 * 2 * 4 - 1, 0);
    CHECK_THROWS_AS(write_png(path, Extent{3, 2}, rgba), PngError);
    CHECK_THROWS_AS(write_png(path, Extent{0, 2}, {}), PngError);
    CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("a path that cannot be written is an error") {
    const std::vector<std::uint8_t> rgba(4, 0);
    CHECK_THROWS_AS(write_png("/no/such/directory/x.png", Extent{1, 1}, rgba), PngError);
}
