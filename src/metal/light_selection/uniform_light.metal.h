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

#include <metal_stdlib>

#include "core/lights/light.h"

namespace serenity {
namespace shaders {

struct UniformLight {
    device const serenity::lights::LightRecord* records;
    uint count;  // at least 1: an estimator in a scene with no light does not select
};

struct SelectedLight {
    serenity::lights::LightRecord light;
    float probability;  // 1 / count
};

// The light `u`, a number in [0, 1), chooses.
inline SelectedLight select_light(UniformLight selection, float u) {
    const uint i = metal::min(uint(u * float(selection.count)), selection.count - 1u);
    return SelectedLight{selection.records[i], 1.0f / float(selection.count)};
}

}  // namespace shaders
}  // namespace serenity
