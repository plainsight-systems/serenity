#pragma once

// A denoiser fixture (training/denoise_check.py, "fixture"): its sizes,
// camera axes and jitters, and its frames' files, read on the CPU, apart
// from any backend. Each backend's runner (<denoise_runner.h>, found by the
// include path, SF.12) runs it through its denoise pass. For the match
// against PyTorch: tests/gpu/denoise_test.cpp on the committed fixture,
// and tests/gpu/denoise_check.cpp on a real network's frames.

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/contracts/linear_image.h"
#include "core/denoise/json.h"
#include "core/denoise/network.h"
#include "core/frame/frame_inputs.h"

namespace serenity::tests {

struct DenoiseFixture {
  std::filesystem::path dir;
  frame::Extent light;
  std::size_t frames = 0;
  contracts::Float3 right{};
  contracts::Float3 up{};
  contracts::Float3 backward{};
  // Each frame's camera jitter (x, y), in light pixels (contract 18).
  std::vector<std::array<float, 2>> jitters;

  std::filesystem::path file(std::size_t k, std::string_view name) const {
    return dir /
           ("frame-" + std::to_string(k) + "-" + std::string(name) + ".pfm");
  }
};

inline DenoiseFixture read_denoise_fixture(const std::filesystem::path& dir) {
  std::ifstream file{dir / "fixture.json", std::ios::binary};
  if (!file) {
    throw std::runtime_error("cannot read " + (dir / "fixture.json").string());
  }
  const std::string text{std::istreambuf_iterator<char>(file),
                         std::istreambuf_iterator<char>()};
  const denoise::Json doc = denoise::parse_json(text);
  const auto count = [](const denoise::Json& node) {
    return static_cast<std::uint32_t>(node.number("a count"));
  };
  const auto vector = [](const denoise::Json& node) {
    const denoise::JsonArray& a = node.array("an axis");
    return contracts::Float3{static_cast<float>(a.at(0).number("x")),
                             static_cast<float>(a.at(1).number("y")),
                             static_cast<float>(a.at(2).number("z"))};
  };
  DenoiseFixture f;
  f.dir = dir;
  const denoise::JsonArray& light = doc.at("light").array("light");
  f.light = {count(light.at(0)), count(light.at(1))};
  f.frames = count(doc.at("frames"));
  f.right = vector(doc.at("axes").at("right"));
  f.up = vector(doc.at("axes").at("up"));
  f.backward = vector(doc.at("axes").at("backward"));
  const denoise::JsonArray& jitters = doc.at("jitter").array("jitter");
  if (jitters.size() != f.frames) {
    throw std::runtime_error(
        "denoise fixture: a jitter a frame, and fixture.json has another "
        "count");
  }
  for (const denoise::Json& j : jitters) {
    const denoise::JsonArray& xy = j.array("a jitter");
    f.jitters.push_back({static_cast<float>(xy.at(0).number("x")),
                         static_cast<float>(xy.at(1).number("y"))});
  }
  return f;
}

}  // namespace serenity::tests
