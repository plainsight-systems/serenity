#pragma once

// Axis: Light selection (uniform).
//
// The naive estimator's selection: one light, each with probability 1 / N,
// from one number. Its cost is the same for ten lights or ten thousand
// (logical-overview.md, principle 9); its noise grows with them, because a
// light chosen is rarely the one that matters at a point. ReSTIR's candidates
// and reuse are what bring that noise down (milestones 2 and 3), and this is
// what they are measured against.

#include <metal_stdlib>

#include "core/lights/sphere_light.h"

namespace serenity {
namespace shaders {

struct UniformLight {
    device const serenity::lights::SphereLightData* lights;
    uint count;
};

struct SelectedLight {
    serenity::lights::SphereLightData light;
    float probability;  // 1 / count
};

// The light `u`, a number in [0, 1), chooses. The scene has at least one
// light: an estimator with none skips aiming.
inline SelectedLight select_light(UniformLight selection, float u) {
    const uint i = metal::min(uint(u * float(selection.count)), selection.count - 1u);
    return SelectedLight{selection.lights[i], 1.0f / float(selection.count)};
}

// The probability with which select_light() chooses any one light: what a
// path that reached a light by sampling a BSDF divides by to weigh itself
// against aiming.
inline float selection_probability(UniformLight selection) {
    return 1.0f / float(selection.count);
}

}  // namespace shaders
}  // namespace serenity
