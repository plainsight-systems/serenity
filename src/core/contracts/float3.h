#pragma once

// Three floats, laid out the same on the host and in shaders: a point, a
// direction or a color in any layout a shader reads (contracts/frame_constants.h
// gives the layout rules). Twelve bytes, no padding. Not the shading
// language's float3, which is sixteen bytes and aligned to sixteen: a shader
// reads one of these and makes a float3 of it.
//
// Data only. Vector arithmetic on the host is done where it is needed, in
// the file whose axis it belongs to.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#endif

namespace serenity {
namespace contracts {

struct Float3 {
    float x;
    float y;
    float z;
};

static_assert(sizeof(Float3) == 12, "Float3 must be the same 12 bytes on the host and in shaders");

}  // namespace contracts
}  // namespace serenity
