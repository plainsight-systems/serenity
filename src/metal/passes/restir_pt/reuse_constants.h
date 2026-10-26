#pragma once

// Axis: Pass (ReSTIR PT's passes): the run's reuse settings as function
// constants.
//
// The path reuse settings (core/contracts/path_reuse_settings.h) are fixed
// for a run: the graph's, made once by the renderer (frame/renderer.cpp,
// make_restir_pt()). So each ReSTIR PT kernel is specialized by them when
// its pipeline is built, as Blender Cycles specializes its kernels on the
// M3 by its scene's KernelData (docs/research/2026-10-23-apple-silicon-
// design.md, section 6, change 1): the compiler folds every comparison and
// loop bound they set, where a kernel read them from a buffer before
// (Per.11: computation moved from run time to compile time). Every field is
// a constant, sigma too, though only the host reads it, so the settings the
// kernels hold are the run's, whole.
//
// What is not specialized, stated: Cycles' feature flags (kernel_features:
// a scene with no volumes, no caustics) have no counterpart here, the demo
// scene using every feature its kernels have, so such flags would fold
// nothing; the frame's own constants (the time, the camera) change every
// frame.
//
// Each field's function-constant index, named once and read by both sides:
// the constants' declarations (reuse_constants.metal.h) and the host's
// values (reuse_constants(), below), so they cannot drift apart (ES.45).
// One metallib holds every pass's kernels, so these are the library's only
// function constants: a new set takes indices after these.

#include "core/contracts/shared_layout.h"

#if !defined(__METAL_VERSION__)
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/contracts/path_reuse_settings.h"
#endif

// NOLINTBEGIN(cppcoreguidelines-use-enum-class): [[function_constant(n)]]
// and Metal's setter take an integer index (Enum.3 deviation, AGENTS.md).
namespace serenity {
namespace bindings {
namespace reuse_constant {
enum Index : uint8_t {
  nee_candidates = 0,
  max_surfaces = 1,
  footprint = 2,
  roughness = 3,
  neighbors = 4,
  sigma = 5,
  confidence_cap = 6,
  cap_floor = 7,
  cap_exponent = 8,
  checkerboard = 9,
};
}  // namespace reuse_constant
}  // namespace bindings
}  // namespace serenity
// NOLINTEND(cppcoreguidelines-use-enum-class)

#if !defined(__METAL_VERSION__)
namespace serenity::metal {

// `settings` as the values of the reuse constants, every one, for a ReSTIR
// PT kernel's pipeline (PipelineOptions::constants, device/library.h).
NS::SharedPtr<MTL::FunctionConstantValues> reuse_constants(
    const contracts::PathReuseSettings& settings);

}  // namespace serenity::metal
#endif
