#pragma once

// Contract 10: a transform. Owned by Shape; read by Animation, Light and
// Acceleration.
//
// The transform that places a shape's geometry, defined in its own
// coordinates (object space), in the world (shapes/primitive.h). Animation
// writes it for a shape that moves (animation/animate.h), a sphere light
// reads where it is and how big from it (lights/sphere_light.h), and the
// acceleration structure places its instances by it, so the families meet
// here and depend on nothing of each other's. Shared with shaders, and the
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
// family's world-space tests say where they rely on it.

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

struct Transform {
    float m[3][4];
};

static_assert(sizeof(Transform) == 48, "Transform must be the same 48 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
// The transform that scales by `scale` (> 0) and then moves by `translation`.
inline Transform placed(Float3 translation, float scale) {
    return Transform{{
        {scale, 0.0f, 0.0f, translation.x},
        {0.0f, scale, 0.0f, translation.y},
        {0.0f, 0.0f, scale, translation.z},
    }};
}

inline Float3 translation(const Transform& t) {
    return {t.m[0][3], t.m[1][3], t.m[2][3]};
}

// `t` with its translation replaced: what a motion changes
// (animation/animate.h).
inline Transform moved_to(Transform t, Float3 translation) {
    t.m[0][3] = translation.x;
    t.m[1][3] = translation.y;
    t.m[2][3] = translation.z;
    return t;
}
#endif

}  // namespace contracts
}  // namespace serenity
