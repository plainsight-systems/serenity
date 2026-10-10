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
    const bool x_wider = metal::abs(s.x) > metal::abs(s.y);
    const float r = x_wider ? s.x : s.y;
    const float phi =
        x_wider ? (M_PI_F / 4.0f) * (s.y / s.x) : (M_PI_F / 2.0f) - (M_PI_F / 4.0f) * (s.x / s.y);
    return r * float2(metal::cos(phi), metal::sin(phi));
}

// Two unit vectors that, with a unit normal n, make a right-handed
// orthonormal basis (t, b, n).
struct Tangents {
    float3 t;
    float3 b;
};

// The tangents of unit `n` (Duff et al., "Building an Orthonormal Basis,
// Revisited", 2017).
inline Tangents tangents(float3 n) {
    const float sign = metal::copysign(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    return Tangents{float3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x),
                    float3(c, sign + n.y * n.y * a, -n.y)};
}

// A direction about unit `n`, distributed by the cosine to it (Malley's
// method), from a point in the unit square.
inline float3 cosine_direction(float3 n, float2 u) {
    const float2 d = concentric_disk(u);
    const Tangents f = tangents(n);
    return d.x * f.t + d.y * f.b + metal::sqrt(metal::max(0.0f, 1.0f - metal::dot(d, d))) * n;
}

}  // namespace shaders
}  // namespace serenity
