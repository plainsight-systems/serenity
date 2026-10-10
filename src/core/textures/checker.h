#pragma once

// Axis: Texture (checker).
//
// Two colors in alternating squares of side `size`, laid on the world's x-z
// plane: the square holding point p is color a if floor(p.x / size) +
// floor(p.z / size) is even, b if odd. The shader half is
// metal/textures/checker.metal.h.

#include "core/contracts/shared_layout.h"
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
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<CheckerData>, "CheckerData is written to the GPU as bytes");
#endif

}  // namespace textures
}  // namespace serenity
