#pragma once

// Axis: GPU backend (shaders).
//
// Reading the core's shared layouts in a shader: a contracts::Float3 is
// twelve packed bytes, the shading language's float3 sixteen, so a shader
// makes one from the other where it reads or writes it (contracts/float3.h).

#include <metal_stdlib>

#include "core/contracts/float3.h"

namespace serenity {
namespace shaders {

inline float3 to_float3(serenity::contracts::Float3 a) {
    return float3(a.x, a.y, a.z);
}

inline serenity::contracts::Float3 to_packed(float3 a) {
    return serenity::contracts::Float3{a.x, a.y, a.z};
}

}  // namespace shaders
}  // namespace serenity
