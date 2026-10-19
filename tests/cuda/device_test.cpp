// The CUDA device: acquired, the architecture the kernels are built for,
// and named, so a green run shows which GPU it ran on.
//
// Not tested here: refusal of a device of another compute capability. The
// one machine these tests run on is the RTX 3090 Ti, so none can show it;
// the refusal is the one branch in cuda/device/device.cpp that says so.

#include "cuda/device/device.h"

#include <doctest/doctest.h>

TEST_CASE("a CUDA device of compute capability 8.6 is acquired") {
  const serenity::cuda::Device device;
  MESSAGE("device: " << device.name());
  CHECK(device.name().find("compute capability 8.6") != std::string::npos);
}
