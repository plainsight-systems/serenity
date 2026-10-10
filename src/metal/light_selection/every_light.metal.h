#pragma once

// Axis: Light selection (every light).
//
// The selection a reference makes: every light, each with probability 1, so
// a pixel's estimate is a sum over the lights and has no selection noise.
// Its cost grows with the number of lights, which is why it is the
// preview's and no estimator's (logical-overview.md, principle 9): the
// estimators choose a fixed number of candidates, and ReSTIR reuses them.
//
// It walks the light records (core/lights/light.h) and names no light kind;
// what each light gives is the emitter's (contract 3,
// metal/lights/emitter.metal.h), as for uniform selection.

#include <metal_stdlib>

#include "core/lights/light.h"

namespace serenity {
namespace shaders {

struct EveryLight {
    constant serenity::lights::LightRecord* records;
    uint count;
};

inline uint selected_count(EveryLight selection) {
    return selection.count;
}

inline serenity::lights::LightRecord selected(EveryLight selection, uint i) {
    return selection.records[i];
}

// The probability with which light `i` was selected.
inline float selection_probability(EveryLight, uint) {
    return 1.0f;
}

}  // namespace shaders
}  // namespace serenity
