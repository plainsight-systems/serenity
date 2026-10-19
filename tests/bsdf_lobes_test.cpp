// Which surfaces an estimator lights by aiming at a light
// (core/contracts/bsdf.h, aims_at_lights): those with a lobe that is not
// delta. A unit of the contract, computed on the CPU and in shaders alike;
// the shader's lobes for each kind are tests/gpu/bsdf_test.cpp's.

#include <doctest/doctest.h>

#include "core/contracts/bsdf.h"

namespace serenity::contracts {
namespace {

TEST_CASE("a surface aims at lights exactly when a lobe of it is not delta") {
  CHECK_FALSE(aims_at_lights(0u));
  CHECK(aims_at_lights(lobe_reflection | lobe_diffuse));      // Lambert
  CHECK(aims_at_lights(lobe_reflection | lobe_glossy));       // rough metal
  CHECK_FALSE(aims_at_lights(lobe_reflection | lobe_delta));  // a mirror
  CHECK_FALSE(aims_at_lights(lobe_reflection | lobe_transmission |
                             lobe_delta));  // glass
  // A coated surface: its color, beside its coat's mirror.
  CHECK(aims_at_lights(lobe_reflection | lobe_diffuse | lobe_delta));
  // A mirror over a glossy coat.
  CHECK(aims_at_lights(lobe_reflection | lobe_delta | lobe_glossy));
}

}  // namespace
}  // namespace serenity::contracts
