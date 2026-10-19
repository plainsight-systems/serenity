// denoise_check FIXTURE OUT: a denoiser fixture (training/denoise_check.py,
// "fixture") run through the GPU's denoise pass, each frame's output
// written to OUT as frame-K-gpu.pfm, for training/denoise_check.py's
// "compare". The check of a real network on real frames
// (docs/research/2026-10-15-denoiser-live.md, "The match against
// PyTorch"); not a ctest test, since its frames and network live beside the
// checkout, not in it. tests/gpu/denoise_test.cpp is the test, on the
// committed fixture.

#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/output/pfm.h"
#include "gpu/support/denoise_fixture.h"

int main(int argc, char** argv) {
  using namespace serenity;
  if (argc != 3) {
    std::cerr << "usage: denoise_check FIXTURE OUT\n";
    return 2;
  }
  try {
    const std::vector<char*> args(argv, argv + argc);
    const tests::DenoiseFixture fixture = tests::read_denoise_fixture(args[1]);
    const std::filesystem::path out = args[2];
    std::filesystem::create_directories(out);
    const std::vector<contracts::LinearImage> outputs =
        tests::run_denoise_fixture(fixture);
    for (std::size_t k = 0; k < outputs.size(); ++k) {
      const std::filesystem::path path =
          out / ("frame-" + std::to_string(k) + "-gpu.pfm");
      output::write_pfm(path, outputs[k]);
      std::cout << path.string() << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "denoise_check: " << error.what() << '\n';
    return 1;
  }
}
