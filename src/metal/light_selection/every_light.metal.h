#pragma once

// Axis: Light selection (every light).
//
// The selection a reference makes: every light, each with probability 1, so
// a pixel's estimate is a sum over the lights and has no selection noise.
// Its cost grows with the number of lights, which is why it is the
// preview's and no estimator's (logical-overview.md, principle 9): the
// estimators choose a fixed number of candidates, and ReSTIR reuses them.
//
// It walks the sphere lights' array itself, where uniform selection chooses
// among light records (uniform_light.metal.h): the preview's direct
// integrator uses the sphere light's exact irradiance, not the emitter
// contract. Both move onto the contracts together, with the preview's
// material switch (docs/process/QUEUE.md).

#include <metal_stdlib>

#include "core/lights/sphere_light.h"

namespace serenity {
namespace shaders {

struct EveryLight {
    device const serenity::lights::SphereLightData* lights;
    uint count;
};

inline uint selected_count(EveryLight selection) {
    return selection.count;
}

inline serenity::lights::SphereLightData selected(EveryLight selection, uint i) {
    return selection.lights[i];
}

// The probability with which light `i` was selected.
inline float selection_probability(EveryLight, uint) {
    return 1.0f;
}

}  // namespace shaders
}  // namespace serenity
