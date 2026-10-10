#pragma once

// Three floats, laid out the same on the host and in shaders: a point, a
// direction or a color in any layout a shader reads (contracts/frame_constants.h
// gives the layout rules). Twelve bytes, no padding. Not the shading
// language's float3, which is sixteen bytes and aligned to sixteen: a shader
// reads one of these and makes a float3 of it.
//
// Data, and on the host one accessor, component(), for the loops over the
// three axes every family writes (ES.3). Vector arithmetic on the host is
// done where it is needed, in the file whose axis it belongs to, at the
// precision that file needs: families depend on contracts, not on each
// other (architecture/file-mapping.md).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

struct Float3 {
    float x;
    float y;
    float z;
};

static_assert(sizeof(Float3) == 12, "Float3 must be the same 12 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<Float3>, "Float3 is written to the GPU as bytes");

// Coordinate `axis` of `v`: x for 0, y for 1, z for 2.
constexpr float component(Float3 v, int axis) noexcept {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}
#endif

}  // namespace contracts
}  // namespace serenity
