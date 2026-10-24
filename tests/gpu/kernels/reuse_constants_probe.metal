// Writes the run's path reuse settings as a ReSTIR PT kernel holds them
// (metal/passes/restir_pt/reuse_constants.metal.h, reuse_settings()), for
// tests/gpu/reuse_constants_test.cpp: its pipeline is specialized by the
// test's settings, so what it writes is what the constants carried. One
// thread. Bindings: the output at 0, one PathReuseSettings.

#include <metal_stdlib>

#include "core/contracts/path_reuse_settings.h"
#include "metal/passes/restir_pt/reuse_constants.metal.h"

kernel void reuse_constants_probe(
    device serenity::contracts::PathReuseSettings* out [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
  if (i == 0u) {
    out[0] = serenity::shaders::reuse_settings();
  }
}
