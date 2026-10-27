// The denoiser's network on the Neural Engine (metal/passes/denoise/
// neural_engine.h), what it refuses: a directory with no Core ML model,
// naming the export that makes one; and a grid that is not one of the
// model's shapes. What it computes is the denoise pass's test against
// PyTorch (denoise_test.cpp); its hand-off, the submission's
// (hand_off_test.cpp).

#include "metal/passes/denoise/neural_engine.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "core/contracts/denoise.h"
#include "core/denoise/network.h"
#include "metal/device/device.h"
#include "metal/device/error.h"
#include "metal/device/submission.h"
#include "support/text.h"
#include "test_paths.h"

namespace contracts = serenity::contracts;
namespace denoise = serenity::denoise;
namespace frame = serenity::frame;
namespace metal = serenity::metal;
namespace tests = serenity::tests;

TEST_CASE("a network with no Core ML model is refused, naming its export") {
  metal::Device device;
  metal::Submission submission(device);
  const std::filesystem::path empty =
      std::filesystem::temp_directory_path() / "serenity-no-model";
  std::filesystem::create_directories(empty);
  const std::string error = tests::error_of<metal::MetalError>([&] {
    return metal::NeuralEngineNetwork(device, submission, empty,
                                      contracts::network_inputs, 8);
  });
  CHECK(error.find("training/export_coreml.py") != std::string::npos);
}

TEST_CASE("the fixture's model has the tests' shapes, and no other is taken") {
  metal::Device device;
  metal::Submission submission(device);
  const std::filesystem::path dir = tests::data_dir / "denoise" / "network";
  const denoise::Network network = denoise::load_network(dir);
  metal::NeuralEngineNetwork engine(device, submission, dir,
                                    contracts::network_inputs,
                                    network.widths.dec.back());
  CHECK_FALSE(engine.rungs().empty());
  const auto row_bytes = [](frame::Extent e) {
    return std::size_t{contracts::tensor_pitch(e.width)} *
           sizeof(std::uint16_t);
  };
  CHECK_THROWS_AS(
      engine.prepare(frame::Extent{4096, 4096}, row_bytes({4096, 4096})),
      metal::MetalError);
  // Each of its shapes is made, Core ML answering in our memory.
  for (const frame::Extent& rung : engine.rungs()) {
    INFO(frame::to_string(rung));
    CHECK_NOTHROW(engine.prepare(rung, row_bytes(rung)));
    CHECK(engine.input() != 0);
    CHECK(engine.features() != 0);
  }
}
