#pragma once

// Axis: Pass (ReSTIR PT's passes): the reuse constants, the shader half of
// reuse_constants.h, which says why the run's settings are function
// constants and gives each its index.
//
// Each declared once, at program scope, as Metal declares a function
// constant; reuse_settings() gives them back as the settings struct the
// integrator's functions take, so a kernel builds it once, at its top, and
// nothing below the entry points changes (the probes' kernels, tests/gpu/
// kernels/, pass their own settings as before).

#include <metal_stdlib>

#include "core/contracts/path_reuse_settings.h"
#include "metal/passes/restir_pt/reuse_constants.h"

namespace serenity {
namespace shaders {

namespace reuse_constant = serenity::bindings::reuse_constant;

constant uint nee_candidates_constant
    [[function_constant(reuse_constant::nee_candidates)]];
constant uint max_surfaces_constant
    [[function_constant(reuse_constant::max_surfaces)]];
constant float footprint_constant
    [[function_constant(reuse_constant::footprint)]];
constant float roughness_constant
    [[function_constant(reuse_constant::roughness)]];
constant uint neighbors_constant
    [[function_constant(reuse_constant::neighbors)]];
constant float sigma_constant [[function_constant(reuse_constant::sigma)]];
constant float confidence_cap_constant
    [[function_constant(reuse_constant::confidence_cap)]];
constant float cap_floor_constant
    [[function_constant(reuse_constant::cap_floor)]];
constant float cap_exponent_constant
    [[function_constant(reuse_constant::cap_exponent)]];
constant uint checkerboard_constant
    [[function_constant(reuse_constant::checkerboard)]];

// The run's settings, every field its constant's value.
inline serenity::contracts::PathReuseSettings reuse_settings() {
  serenity::contracts::PathReuseSettings settings{};
  settings.nee_candidates = nee_candidates_constant;
  settings.max_surfaces = max_surfaces_constant;
  settings.footprint = footprint_constant;
  settings.roughness = roughness_constant;
  settings.neighbors = neighbors_constant;
  settings.sigma = sigma_constant;
  settings.confidence_cap = confidence_cap_constant;
  settings.cap_floor = cap_floor_constant;
  settings.cap_exponent = cap_exponent_constant;
  settings.checkerboard = checkerboard_constant;
  return settings;
}

}  // namespace shaders
}  // namespace serenity
