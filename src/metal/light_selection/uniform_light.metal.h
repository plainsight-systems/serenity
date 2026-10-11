#pragma once

// Axis: Light selection (uniform).
//
// The naive estimator's selection: one light, each with probability 1 / N,
// from one number. It chooses among light records (core/lights/light.h) and
// names no light kind; what a chosen light gives is the emitter's (contract
// 3, metal/lights/emitter.metal.h). Its cost is the same for ten lights or
// ten thousand (logical-overview.md, principle 9); its noise grows with
// them, because a light chosen is rarely the one that matters at a point.
// ReSTIR's candidates and reuse are what bring that noise down (milestones 2
// and 3), and this is what they are measured against.
//
// And the probability with which it chooses a given light, light_probability()
// (pbrt-v4's LightSampler::PMF): what the path tracer's MIS weighs a light
// struck by a bounced ray by, as the chance step 5 had of choosing it
// (integrator/path.metal.h, step 3). 1 / N for every light, the light named
// so that a strategy whose chance differs by light answers by the same
// signature. Asked only of a light that exists, so N is at least 1 (I.5).

#include <metal_stdlib>

#include "core/lights/light.h"

namespace serenity {
namespace shaders {

struct UniformLight {
    constant serenity::lights::LightRecord* records;
    uint count;  // 0 in a scene with no light, which select_light() must not be asked of
};

struct SelectedLight {
    serenity::lights::LightRecord light;
    float probability;  // 1 / count
};

// The light `u`, a number in [0, 1), chooses. Precondition: count is at
// least 1; the estimator tests it before asking (path.metal.h, step 5).
inline SelectedLight select_light(UniformLight selection, float u) {
    const uint i = metal::min(uint(u * float(selection.count)), selection.count - 1u);
    return SelectedLight{selection.records[i], 1.0f / float(selection.count)};
}

// The probability select_light() chooses `light` with (above). Precondition:
// count is at least 1, as it is wherever a light exists to be asked of.
inline float light_probability(UniformLight selection, serenity::lights::LightRecord) {
    return 1.0f / float(selection.count);
}

// For ReSTIR DI's candidates, which keep a light by its record's index
// (contracts/light_sample.h): the index select_light() chooses for the same
// `u`, by the same formula, and its probability. select_light() stays as it
// is, so the naive path tracer's image does not move by a bit; the formula
// is written twice, three lines, rather than select_light() rewritten over
// this one, which would put the path tracer's image at the compiler's
// mercy for no gain (ES.3 weighed against the additive rule).
struct SelectedIndex {
    uint index;         // into the light records
    float probability;  // 1 / count
};
inline SelectedIndex select_light_index(UniformLight selection, float u) {
    const uint i = metal::min(uint(u * float(selection.count)), selection.count - 1u);
    return SelectedIndex{i, 1.0f / float(selection.count)};
}

}  // namespace shaders
}  // namespace serenity
