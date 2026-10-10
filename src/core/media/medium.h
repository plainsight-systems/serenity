#pragma once

// Axis: Medium.
//
// The medium kinds, and the record that says which a medium is: a kind and
// an index into that kind's array, as materials are (materials/material.h).
// Shared with shaders. A shape names a medium by the index of its record
// (shapes/primitive.h, ShapeRecord::interior), or contracts::no_medium.
// Each kind answers contract 12 (contracts/medium.h); the one place a shader
// dispatches on MediumKind is metal/media/media.metal.h.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

namespace serenity {
namespace media {

enum class MediumKind : uint32_t {
    absorbing = 0,  // absorbs and does not scatter: tinted glass (absorbing.h)
};

struct MediumRecord {
    MediumKind kind;
    uint32_t index;  // into that kind's array
};

static_assert(sizeof(MediumRecord) == 8, "MediumRecord must be the same 8 bytes on the host and in shaders");

}  // namespace media
}  // namespace serenity
