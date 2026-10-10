#pragma once

// Contract 10: a transform. Owned by Shape; read by Animation, Light and
// Acceleration.
//
// The transform that places a shape's geometry, defined in its own
// coordinates (object space), in the world (shapes/primitive.h). Animation
// writes it for a shape that moves (animation/animate.h), a sphere light
// reads where it is and how big from it (lights/sphere_light.h), and the
// acceleration structure places each shape's box by it, so the families meet
// here and depend on nothing of each other's. Shared with shaders, and the
// same 3 x 4 row-major affine matrix Vulkan's instance descriptors take
// (VkTransformMatrixKHR) and Metal's take when built with
// MTLMatrixLayoutRowMajor, so the bytes could be given to either as they are
// when a shape is a mesh under an instance.
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

#include "core/contracts/shared_layout.h"
#include "core/contracts/float3.h"

namespace serenity {
namespace contracts {

struct Transform {
    float m[3][4];
};

static_assert(sizeof(Transform) == 48, "Transform must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<Transform>, "Transform is written to the GPU as bytes");

// The transform that scales by `factor` (> 0) and then moves by `offset`.
constexpr Transform placed(Float3 offset, float factor) noexcept {
    return Transform{{
        {factor, 0.0f, 0.0f, offset.x},
        {0.0f, factor, 0.0f, offset.y},
        {0.0f, 0.0f, factor, offset.z},
    }};
}

constexpr Float3 translation(const Transform& t) noexcept {
    return {t.m[0][3], t.m[1][3], t.m[2][3]};
}

// The uniform scale of a transform the scene reader made, a translation and
// a scale (above): a sphere's radius (shapes/sphere.h).
constexpr float scale(const Transform& t) noexcept {
    return t.m[0][0];
}

// `t` with its translation replaced by `to`: what a motion changes
// (animation/animate.h).
constexpr Transform moved_to(Transform t, Float3 to) noexcept {
    t.m[0][3] = to.x;
    t.m[1][3] = to.y;
    t.m[2][3] = to.z;
    return t;
}
#endif

}  // namespace contracts
}  // namespace serenity
