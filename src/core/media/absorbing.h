#pragma once

// Axis: Medium (absorbing).
//
// A medium that absorbs light and scatters none: the inside of tinted glass.
// By Beer and Lambert, light crossing a stretch t of it keeps
//
//   transmittance(t) = exp(-absorption t)
//
// per channel (contract 12), so thick glass is deeper in color than thin,
// and a marble's middle deeper than its rim. A scene gives it as the color
// white light keeps through `tint_distance` meters of it (core/scene/
// scene.h), absorption = -ln(tint) / tint_distance, tint in (0, 1] per
// channel. The shader half is metal/media/absorbing.metal.h.

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace media {

struct AbsorbingData {
    contracts::Float3 absorption;  // per meter, each 0 or more, finite
    uint32_t padding;
};

static_assert(sizeof(AbsorbingData) == 16, "AbsorbingData must be the same 16 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<AbsorbingData>, "AbsorbingData is written to the GPU as bytes");
#endif

}  // namespace media
}  // namespace serenity
