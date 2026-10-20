// The headless program on the CUDA backend, as its user runs it, an
// integration test: the test pattern written as PNGs, each the CPU's
// pattern at its frame's time, red and green to the bit and blue within a
// level (the GPU's sinf and the CPU's sin may differ in the last bit); the
// reference graph, the path tracer through the tone map, the same command
// writing the same bytes; and a graph with a pass the backend does not
// build refused by name, before any directory is made. It names the
// machine it ran on, as every GPU verification does. What the program's
// runs and refusals print on any backend is tests/test_headless.sh's.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "cuda/device/device.h"
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

// stb's decoded pixels, freed by stb's own function however the test ends
// (R.1, R.12).
using Decoded = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;

TEST_CASE("the headless program writes the test pattern on CUDA") {
  MESSAGE("device: " << serenity::cuda::Device().name());
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
    const Decoded decoded(stbi_load(png.string().c_str(), &w, &h, &channels, 4),
                          &stbi_image_free);
    REQUIRE(decoded != nullptr);
    // The decoded size first: stb's buffer holds w x h x 4 bytes, and is read
    // only once that is the size asked for (SL.con.3).
    REQUIRE(w == 40);
    REQUIRE(h == 24);
    const std::span<const stbi_uc> bytes(decoded.get(),
                                         std::size_t{40} * 24 * 4);
    const std::vector<std::uint8_t> pixels(bytes.begin(), bytes.end());
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
        CHECK(pixels.at(at) == want[0]);
        CHECK(pixels.at(at + 1) == want[1]);
        worst_blue =
            std::max(worst_blue, std::abs(pixels.at(at + 2) - want[2]));
        CHECK(pixels.at(at + 3) == 255);
      }
    }
    CHECK(worst_blue <= 1);
  }
}

TEST_CASE("the reference graph writes the same bytes from the same command") {
  const ScratchDirectory scratch("cuda-headless-path");
  const std::filesystem::path graph = scratch.path() / "path.toml";
  serenity::tests::write_bytes(
      graph,
      "passes = [\"path\", \"tone_map\"]\n[tone_map]\nexposure = 0.0\n"
      "bloom = 0.04\n");
  // Twice, into two directories: a moving scene, so the structures are
  // rebuilt and the accumulated image starts over every frame.
  std::array<std::string, 2> written;
  for (std::size_t run = 0; run < written.size(); ++run) {
    const std::filesystem::path out =
        scratch.path() / ("out" + std::to_string(run));
    const Ran ran = serenity::tests::run_program(
        serenity::tests::headless_program,
        {"--graph", graph.string(), "--scene",
         (serenity::tests::scenes_dir / "brass_sphere_flight.toml").string(),
         "--out", out.string(), "--size", "48x32", "--frames", "3", "--samples",
         "2", "--write", "last"},
        scratch.path());
    REQUIRE(ran.status == 0);
    CHECK(ran.err.empty());
    const std::filesystem::path png = out / "frame-000002.png";
    REQUIRE(std::filesystem::exists(png));
    written.at(run) = serenity::tests::read_bytes(png);
  }
  CHECK_FALSE(written[0].empty());
  CHECK(written[0] == written[1]);
}

TEST_CASE("a graph the CUDA backend cannot render yet is refused by name") {
  const ScratchDirectory scratch("cuda-headless-refused");
  const std::filesystem::path graph = scratch.path() / "features.toml";
  serenity::tests::write_bytes(graph, "passes = [\"features\"]\n");
  const std::filesystem::path out = scratch.path() / "out";
  const Ran ran = serenity::tests::run_program(
      serenity::tests::headless_program,
      {"--graph", graph.string(), "--scene",
       (serenity::tests::scenes_dir / "brass_sphere.toml").string(), "--out",
       out.string(), "--size", "16x8", "--features"},
      scratch.path());
  CHECK(ran.status == 1);
  CHECK(serenity::tests::contains(ran.err, "has no features pass yet"));
  CHECK_FALSE(std::filesystem::exists(out));
}

}  // namespace
