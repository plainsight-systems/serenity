#pragma once

// Axis: Shape (where an instance of a geometry is).
//
// The transform that places a shape's geometry, defined in its own
// coordinates (object space), in the world: shared with shaders, and the
// same 3 x 4 row-major affine matrix Vulkan's instance descriptors take
// (VkTransformMatrixKHR) and Metal's take when built with
// MTLMatrixLayoutRowMajor, so the bytes are given to either as they are.
// Row r is (m[r][0], m[r][1], m[r][2], m[r][3]): the linear part in the
// first three columns, the translation in the fourth.
//
// Every transform here is a similarity: a rotation, a uniform scale s > 0,
// and a translation. That is what keeps the rest simple and exact:
//
//   - the normal turns by the rotation alone, so object space's normal,
//     rotated, is the world's, with no inverse-transpose;
//   - a sphere stays a sphere, so a sphere light's center is the
//     translation and its radius the scale (lights/sphere_light.h);
//   - a ray's parameter t is the same in both spaces: Metal and Vulkan carry
//     the world's ray into object space without renormalizing its
//     direction, so an object-space hit at t is the world's hit at t.
//
// The scene format has no rotation yet: every transform the scene reader
// makes is a translation and a scale (core/scene/scene.h), and the Shape
// family's world-space helpers say where they rely on it.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace shapes {

struct Transform {
    float m[3][4];
};

static_assert(sizeof(Transform) == 48, "Transform must be the same 48 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// The transform that scales by `scale` (> 0) and then moves by `translation`.
inline Transform placed(contracts::Float3 translation, float scale) {
    return Transform{{
        {scale, 0.0f, 0.0f, translation.x},
        {0.0f, scale, 0.0f, translation.y},
        {0.0f, 0.0f, scale, translation.z},
    }};
}

inline contracts::Float3 translation(const Transform& t) {
    return {t.m[0][3], t.m[1][3], t.m[2][3]};
}

// `t` with its translation replaced: what a motion changes (core/scene/animate.h).
inline Transform moved_to(Transform t, contracts::Float3 translation) {
    t.m[0][3] = translation.x;
    t.m[1][3] = translation.y;
    t.m[2][3] = translation.z;
    return t;
}
#endif

}  // namespace shapes
}  // namespace serenity
