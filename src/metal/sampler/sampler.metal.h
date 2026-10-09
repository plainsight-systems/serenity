#pragma once

// Axis: Sampler.
//
// Numbers in [0, 1) derived from where and why they are used: the pixel, the
// purpose (a light, the sky, a reflection), and the sample's index. There is
// no generator state (logical-overview.md, principle 2), so any pixel can be
// rendered again exactly, alone.
//
// A pixel's samples for one purpose are the R2 sequence (Roberts' additive
// recurrence on the plastic constant), shifted by an offset hashed from the
// pixel and the purpose (Cranley-Patterson rotation): evenly spread within
// the pixel's set, and uncorrelated between neighbouring pixels, so what
// error is left is fine grain rather than bands. The frame is not an input
// yet: the preview takes no average across frames, and a still scene stays
// still. When frames are averaged, the frame index joins the hash.
//
// The hash is PCG's output permutation (Jarzynski and Olano, "Hash Functions
// for GPU Rendering", 2020).

#include <metal_stdlib>

namespace serenity {
namespace shaders {

inline uint pcg_hash(uint v) {
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float unit_float(uint bits) {
    return float(bits >> 8) * (1.0f / 16777216.0f);
}

// The offset of `pixel`'s samples for `purpose`.
inline float2 sample_offset(uint2 pixel, uint purpose) {
    const uint h = pcg_hash(pixel.x ^ pcg_hash(pixel.y ^ pcg_hash(purpose)));
    return float2(unit_float(h), unit_float(pcg_hash(h)));
}

// Sample `index` of a pixel's set whose offset is `offset`.
inline float2 sample_2d(float2 offset, uint index) {
    const float2 step = float2(0.7548776662466927f, 0.5698402909980532f);  // 1/p, 1/p^2
    return metal::fract(offset + float(index) * step);
}

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
