// The headless program on the CUDA backend, as its user runs it, an
// integration test: the test pattern written as PNGs, each the CPU's
// pattern at its frame's time, blue within a level; and a graph with a pass
// the backend does not build refused by name, before any directory is made.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "headless_program.h"
#include "support/files.h"
#include "support/run_program.h"
#include "support/text.h"
#include "test_paths.h"
#include "test_pattern_reference.h"

// stb_image's implementation, in this translation unit alone, to decode
// what the program wrote.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

namespace {

using serenity::tests::Ran;
using serenity::tests::ScratchDirectory;

TEST_CASE("the headless program writes the test pattern on CUDA") {
  const ScratchDirectory scratch("cuda-headless");
  const std::filesystem::path graph = scratch.path() / "pattern.toml";
  serenity::tests::write_bytes(graph, "passes = [\"test_pattern\"]\n");
  const std::filesystem::path out = scratch.path() / "out";
  const Ran ran = serenity::tests::run_program(
      serenity::tests::headless_program,
      {"--graph", graph.string(), "--out", out.string(), "--size", "40x24",
       "--frames", "2", "--step", "0.75"},
      scratch.path());
  REQUIRE(ran.status == 0);
  CHECK(ran.err.empty());
  for (std::uint64_t index = 0; index < 2; ++index) {
    CAPTURE(index);
    const std::filesystem::path png =
        out / ("frame-00000" + std::to_string(index) + ".png");
    REQUIRE(std::filesystem::exists(png));
    CHECK(serenity::tests::contains(ran.out, png.string()));
    int w = 0;
    int h = 0;
    int channels = 0;
    unsigned char* pixels =
        stbi_load(png.string().c_str(), &w, &h, &channels, 4);
    REQUIRE(pixels != nullptr);
    CHECK(w == 40);
    CHECK(h == 24);
    int worst_blue = 0;
    const float seconds = 0.75f * static_cast<float>(index);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const std::size_t at =
            (static_cast<std::size_t>(y) * 40 + static_cast<std::size_t>(x)) *
            4;
        const std::array<std::uint8_t, 4> want =
            serenity::tests::test_pattern_pixel(static_cast<unsigned>(x),
                                                static_cast<unsigned>(y), 40,
                                                24, seconds);
        CHECK(pixels[at] == want[0]);
        CHECK(pixels[at + 1] == want[1]);
        worst_blue = std::max(worst_blue, std::abs(pixels[at + 2] - want[2]));
        CHECK(pixels[at + 3] == 255);
      }
    }
    CHECK(worst_blue <= 1);
    stbi_image_free(pixels);
  }
}

TEST_CASE("a graph the CUDA backend cannot render yet is refused by name") {
  const ScratchDirectory scratch("cuda-headless-refused");
  const std::filesystem::path graph = scratch.path() / "path.toml";
  serenity::tests::write_bytes(
      graph,
      "passes = [\"path\", \"tone_map\"]\n[tone_map]\nexposure = 0.0\n"
      "bloom = 0.0\n");
  const std::filesystem::path out = scratch.path() / "out";
  const Ran ran = serenity::tests::run_program(
      serenity::tests::headless_program,
      {"--graph", graph.string(), "--scene",
       (serenity::tests::scenes_dir / "brass_sphere.toml").string(), "--out",
       out.string(), "--size", "16x8"},
      scratch.path());
  CHECK(ran.status == 1);
  CHECK(serenity::tests::contains(ran.err, "has no path pass yet"));
  CHECK_FALSE(std::filesystem::exists(out));
}

}  // namespace
