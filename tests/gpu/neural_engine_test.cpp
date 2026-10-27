// The denoiser's network on the Neural Engine (metal/passes/denoise/
// neural_engine.h), what it refuses: a directory with no Core ML model,
// naming the export that makes one; a grid that is not one of the model's
// shapes; and rows not the kernels' pitch, which would break the one copy
// Metal and Core ML share, keeping nothing half made. What it computes is
// the denoise pass's test against
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
  const std::filesystem::path none =
      tests::data_dir / "denoise" / "no-such-network";
  const std::string error = tests::error_of<metal::MetalError>([&] {
    return metal::NeuralEngineNetwork(device, submission, none,
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
  REQUIRE(engine.rungs().size() >= 2);
  const auto row_bytes = [](frame::Extent e) {
    return std::size_t{contracts::tensor_pitch(e.width)} *
           sizeof(std::uint16_t);
  };
  CHECK(tests::error_of<metal::MetalError>([&] {
          engine.prepare(frame::Extent{4096, 4096}, row_bytes({4096, 4096}));
        }) ==
        "NeuralEngineNetwork: 4096 x 4096 is not one of the model's "
        "shapes");
  // Each of its shapes is made, Core ML answering in our memory.
  for (const frame::Extent& rung : engine.rungs()) {
    INFO(frame::to_string(rung));
    CHECK_NOTHROW(engine.prepare(rung, row_bytes(rung)));
    CHECK(engine.prepared() == rung);
  }
  // Rows not the kernels' pitch: at the rung it is ready for, refused and
  // kept; at another, refused by Core Video's own pitch, and none kept.
  const frame::Extent last = engine.rungs().back();
  CHECK(tests::contains(tests::error_of<metal::MetalError>([&] {
                          engine.prepare(last, row_bytes(last) + 64);
                        }),
                        "bytes apart, not"));
  CHECK(engine.prepared() == last);
  const frame::Extent first = engine.rungs().front();
  CHECK(tests::contains(tests::error_of<metal::MetalError>([&] {
                          engine.prepare(first, row_bytes(first) + 64);
                        }),
                        "Core Video pads"));
  CHECK(engine.prepared() == frame::Extent{});
  CHECK(engine.input() == 0);
}
