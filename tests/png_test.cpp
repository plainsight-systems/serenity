// PNG output: a file that decodes to exactly the pixels given, rows from the
// top, R, G, B, A; and a wrong-sized buffer refused before anything is
// written.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

#include "core/output/png.h"
#include "support/files.h"

using serenity::frame::Extent;
using serenity::output::PngError;
using serenity::output::write_png;
using serenity::tests::read_bytes;
using serenity::tests::ScratchDirectory;

namespace {

std::uint32_t big_endian(const std::string& bytes, std::size_t at) {
  const auto byte = [&](std::size_t i) {
    return std::uint32_t{static_cast<unsigned char>(bytes[at + i])};
  };
  return (byte(0) << 24) | (byte(1) << 16) | (byte(2) << 8) | byte(3);
}

// stb's decoded pixels, freed by stb's own function however the test ends
// (R.1, R.12).
using Decoded = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;

}  // namespace

TEST_CASE(
    "a PNG holds exactly the pixels given: its size in the header, each "
    "channel in its place") {
  // A different value in every channel of every pixel: a swapped channel,
  // a row written bottom up, or a wrong stride shows. Not square, so
  // width and height cannot be swapped unseen.
  constexpr Extent size{5, 3};
  std::vector<std::uint8_t> rgba(std::size_t{size.width} * size.height * 4);
  for (std::size_t i = 0; i < rgba.size(); ++i) {
    rgba[i] = static_cast<std::uint8_t>(7 + 3 * i);
  }
  const ScratchDirectory scratch("png-test");
  write_png(scratch / "frame.png", size, rgba);

  const std::string bytes = read_bytes(scratch / "frame.png");
  REQUIRE(bytes.size() > 33);
  constexpr std::array<unsigned char, 8> signature{0x89, 'P',  'N',  'G',
                                                   0x0d, 0x0a, 0x1a, 0x0a};
  CHECK(std::equal(signature.begin(), signature.end(), bytes.begin(),
                   [](unsigned char a, char b) {
                     return a == static_cast<unsigned char>(b);
                   }));
  CHECK(bytes[12] == 'I');
  CHECK(bytes[15] == 'R');
  CHECK(big_endian(bytes, 16) == size.width);
  CHECK(big_endian(bytes, 20) == size.height);

  const std::vector<stbi_uc> encoded(bytes.begin(), bytes.end());
  int width = 0;
  int height = 0;
  int channels = 0;
  const Decoded decoded(
      stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()),
                            &width, &height, &channels, 0),
      &stbi_image_free);
  REQUIRE(decoded != nullptr);
  // The decoded size first: stb's buffer holds width x height x channels
  // bytes, and is read only once that is the size given (SL.con.3).
  REQUIRE(width == static_cast<int>(size.width));
  REQUIRE(height == static_cast<int>(size.height));
  REQUIRE(channels == 4);
  const std::span<const stbi_uc> decoded_bytes(decoded.get(), rgba.size());
  const std::vector<std::uint8_t> pixels(decoded_bytes.begin(),
                                         decoded_bytes.end());
  CHECK(pixels == rgba);
}

TEST_CASE("a buffer of the wrong size is refused, and nothing is written") {
  const ScratchDirectory scratch("png-wrong");
  const std::vector<std::uint8_t> rgba(3 * 2 * 4 - 1, 0);
  CHECK_THROWS_AS(write_png(scratch / "frame.png", Extent{3, 2}, rgba),
                  PngError);
  CHECK_THROWS_AS(write_png(scratch / "frame.png", Extent{0, 2}, {}), PngError);
  CHECK(std::filesystem::is_empty(scratch.path()));
}

TEST_CASE("a path that cannot be written is an error") {
  const std::vector<std::uint8_t> rgba(4, 0);
  CHECK_THROWS_AS(write_png("/no/such/directory/x.png", Extent{1, 1}, rgba),
                  PngError);
}
