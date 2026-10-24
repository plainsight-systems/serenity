// The run's path reuse settings reach a ReSTIR PT kernel whole through its
// function constants (metal/passes/restir_pt/reuse_constants.h): the host's
// values, set by index, and the shader's declarations, read by the same
// index, agree field for field. Each field is given a value no other has,
// so a constant read at another's index, or a field assembled from the
// wrong constant, fails. Metal's alone: the CUDA backend reads its
// settings from memory.

#include <doctest/doctest.h>

#include <vector>

// the backend's runner, by the include path (SF.12)
#include <probe_runner.h>

#include "core/contracts/path_reuse_settings.h"
#include "metal/passes/restir_pt/reuse_constants.h"

namespace contracts = serenity::contracts;
namespace tests = serenity::tests;

TEST_CASE(
    "the reuse constants carry every field of the run's settings to the "
    "kernel, each at its own index") {
  contracts::PathReuseSettings settings{};
  settings.nee_candidates = 3u;
  settings.max_surfaces = 7u;
  settings.footprint = 0.25f;
  settings.roughness = 0.5f;
  settings.neighbors = 2u;
  settings.sigma = 11.0f;
  settings.confidence_cap = 20.0f;
  settings.cap_floor = 1.5f;
  settings.cap_exponent = 0.125f;
  settings.checkerboard = 1u;

  tests::ProbeRunner gpu;
  const auto constants = serenity::metal::reuse_constants(settings);
  const std::vector<contracts::PathReuseSettings> held =
      gpu.run<contracts::PathReuseSettings>("reuse_constants_probe", 1, {},
                                            {.constants = constants.get()});
  REQUIRE(held.size() == 1);
  CHECK(held[0].nee_candidates == settings.nee_candidates);
  CHECK(held[0].max_surfaces == settings.max_surfaces);
  CHECK(held[0].footprint == settings.footprint);
  CHECK(held[0].roughness == settings.roughness);
  CHECK(held[0].neighbors == settings.neighbors);
  CHECK(held[0].sigma == settings.sigma);
  CHECK(held[0].confidence_cap == settings.confidence_cap);
  CHECK(held[0].cap_floor == settings.cap_floor);
  CHECK(held[0].cap_exponent == settings.cap_exponent);
  CHECK(held[0].checkerboard == settings.checkerboard);
}
