#pragma once

// Axis: Texture (checker).
//
// Two colors in alternating squares of side `size`, laid on the world's x-z
// plane: the square holding point p is color a if floor(p.x / size) +
// floor(p.z / size) is even, b if odd. The shader half is
// metal/textures/checker.metal.h.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace textures {

struct CheckerData {
    contracts::Float3 a;
    float size;  // side of a square, in scene units; greater than 0
    contracts::Float3 b;
    uint32_t padding;
};

static_assert(sizeof(CheckerData) == 32, "CheckerData must be the same 32 bytes on the host and in shaders");

}  // namespace textures
}  // namespace serenity
