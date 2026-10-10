#pragma once

// Axis: Shape.
//
// What a shape kind's exact test answers of a ray (shapes.metal.h): whether
// the ray crosses its surface within the ray's (min_distance, max_distance),
// and at what parameter t. One result, returned, rather than a bool and an
// out parameter (F.20, F.21).

#include <metal_stdlib>

namespace serenity {
namespace shaders {

struct Crossing {
    bool found;
    float t;  // the ray's parameter at the crossing; 0 where none was found
};

inline Crossing no_crossing() {
    return Crossing{false, 0.0f};
}

}  // namespace shaders
}  // namespace serenity
