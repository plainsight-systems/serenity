#pragma once

// Axis: Light selection (uniform).
//
// The naive estimator's selection, the path tracer's, the reference's
// (ReSTIR PT's is by power, metal/light_selection/power_light.metal.h): one
// light, each with probability 1 / N, from one number. It chooses among light
// records (core/lights/light.h) and names no light kind; what a chosen light
// gives is the emitter's (contract 3, cuda/lights/emitter.cuh). Its cost is the
// same for ten lights or ten thousand (logical-overview.md, principle 9); its
// noise grows with them, because a light chosen is rarely the one that matters
// at a point. ReSTIR's candidates and reuse are what bring that noise down
// (milestones 2 and 3), and this is what they are measured against.
//
// And the probability with which it chooses a given light, light_probability()
// (pbrt-v4's LightSampler::PMF): what the path tracer's MIS weighs a light
// struck by a bounced ray by, as the chance step 5 had of choosing it
// (integrator/path.cuh, step 3). 1 / N for every light, the light named
// so that a strategy whose chance differs by light answers by the same
// signature. Asked only of a light that exists, so N is at least 1 (I.5).
//
// Translated line for line from Metal's
// metal/light_selection/uniform_light.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/lights/light.h"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

struct UniformLight {
  const serenity::lights::LightRecord* records;
  uint count;  // 0 in a scene with no light, which select_light() must not be
               // asked of
};

struct SelectedLight {
  serenity::lights::LightRecord light;
  float probability;  // 1 / count
};

// The probability a uniform choice takes any one light with: the one place
// this axis's density is written, which every estimator asks rather than
// assume uniform selection itself (I.1). Precondition: count is at least
// 1, as it is wherever a light exists to be asked of.
__device__ inline float selection_probability(UniformLight selection) {
  return 1.0f / float(selection.count);
}

// The index `u`, a number in [0, 1), chooses, and its probability.
// ReSTIR PT draws its tiles by power instead
// (metal/light_selection/power_light.metal.h). Precondition: count is at least
// 1; the estimator tests it before asking (integrator/path.cuh,
// step 5).
struct SelectedIndex {
  uint index;         // into the light records
  float probability;  // 1 / count
};
__device__ inline SelectedIndex select_light_index(UniformLight selection,
                                                   float u) {
  const uint i = min(uint(u * float(selection.count)), selection.count - 1u);
  return SelectedIndex{i, selection_probability(selection)};
}

// The light `u` chooses: the same choice, its record. Built on
// select_light_index(), so the two cannot drift apart (ES.3); the path
// tracer's image is the same to the bit as when the formula was written
// twice (docs/research/2026-10-16-codebase-guidelines-pass.md, C18).
__device__ inline SelectedLight select_light(UniformLight selection, float u) {
  const SelectedIndex chosen = select_light_index(selection, u);
  return SelectedLight{selection.records[chosen.index], chosen.probability};
}

// The probability select_light() chooses `light` with (above).
__device__ inline float light_probability(UniformLight selection,
                                          serenity::lights::LightRecord) {
  return selection_probability(selection);
}

}  // namespace shaders
}  // namespace serenity
