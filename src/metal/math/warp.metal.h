#pragma once

// Shared shader mathematics: an orthonormal frame about a direction, and
// warps from the unit square to the disk and the hemisphere. Mappings, not
// sources of numbers: the sampler (sampler/sampler.metal.h) derives the
// numbers, and each family that turns numbers into a point or a direction
// (a material's sample(), a light's cone, the integrator's sky rays) maps
// them here. So no family reaches into another for a mapping, and the
// sampler can change its sequences without touching one. pbrt keeps the
// same split: its warps in util/sampling.h, apart from its Samplers.

#include <metal_stdlib>

namespace serenity {
namespace shaders {

// A point in the unit disk, area-preserving and continuous (Shirley and
// Chiu's concentric map), from a point in the unit square.
inline float2 concentric_disk(float2 u) {
    const float2 s = 2.0f * u - 1.0f;
    if (s.x == 0.0f && s.y == 0.0f) {
        return float2(0.0f);
    }
    float r;
    float phi;
    if (metal::abs(s.x) > metal::abs(s.y)) {
        r = s.x;
        phi = (M_PI_F / 4.0f) * (s.y / s.x);
    } else {
        r = s.y;
        phi = (M_PI_F / 2.0f) - (M_PI_F / 4.0f) * (s.x / s.y);
    }
    return r * float2(metal::cos(phi), metal::sin(phi));
}

// Two unit vectors that, with unit `n`, make a right-handed orthonormal
// basis (Duff et al., "Building an Orthonormal Basis, Revisited", 2017).
inline void basis(float3 n, thread float3& t, thread float3& b) {
    const float sign = metal::copysign(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    t = float3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x);
    b = float3(c, sign + n.y * n.y * a, -n.y);
}

// A direction about unit `n`, distributed by the cosine to it (Malley's
// method), from a point in the unit square.
inline float3 cosine_direction(float3 n, float2 u) {
    const float2 d = concentric_disk(u);
    float3 t;
    float3 b;
    basis(n, t, b);
    return d.x * t + d.y * b + metal::sqrt(metal::max(0.0f, 1.0f - metal::dot(d, d))) * n;
}

}  // namespace shaders
}  // namespace serenity
