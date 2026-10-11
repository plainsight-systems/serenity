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

// The equal-area octahedral map between the unit sphere and the square
// (Clarberg 2008), quantized to two 16-bit numbers in one word, x in bits
// 0 to 15 and y in 16 to 31 (contracts/light_sample.h states it and what
// it loses): how ReSTIR DI stores a light's point on its unit sphere
// (contract 4), a surface's wo and its shading normal (contracts 5 and 2).
// A mapping every family whose points are unit directions may use, so it
// is here, not in one of them.
//
// Written from pbrt-v4's EqualAreaSphereToSquare and EqualAreaSquareToSphere
// (Clarberg's map), with the square [-1, 1]^2 and one change for precision:
// the radius r = sqrt(1 - |z|) is taken as sqrt((x^2 + y^2) / (1 + |z|)),
// the same number for a unit vector, which does not cancel near the poles,
// where 1 - |z| keeps only float's absolute 6e-8 and r's error would be
// several cells (ES.46: the narrowing stated is the 16 bits, not a float's
// cancellation besides). Each coordinate is then rounded to the nearest of
// 65536 levels, round((c + 1) / 2 x 65535). The round trip's error, the
// angle between a unit vector and its code's decoding, is at most
// contracts::unit_round_trip_error (core/contracts/light_sample.h, where
// the quantization is stated), measured by tests/gpu/restir_test.cpp over
// the whole sphere.
//
// Cost: encode, a square root, an arctangent and a division; decode, a
// square root, a sine and a cosine. Pure (F.8), inline, no memory read.
constant constexpr float unit_levels = 65535.0f;  // 2^16 - 1, the largest code
constant constexpr uint unit_code_mask = 0xffffu;  // a coordinate's code, x's in the low bits
constant constexpr uint unit_y_shift = 16u;        // y's above it

// A coordinate in [-1, 1] as its 16-bit code, and back.
inline uint encode_coordinate(float c) {
    return uint(metal::rint(metal::clamp((c + 1.0f) * 0.5f, 0.0f, 1.0f) * unit_levels));
}

inline float decode_coordinate(uint code) {
    return float(code) * (2.0f / unit_levels) - 1.0f;
}

inline uint encode_unit(float3 unit) {
    const float3 a = metal::abs(unit);
    const float r = metal::sqrt((a.x * a.x + a.y * a.y) / (1.0f + a.z));
    const float wide = metal::max(a.x, a.y);
    const float narrow = metal::min(a.x, a.y);
    float phi = wide > 0.0f ? metal::atan(narrow / wide) * (2.0f / M_PI_F) : 0.0f;
    if (a.x < a.y) {
        phi = 1.0f - phi;
    }
    float v = phi * r;
    float u = r - v;
    if (unit.z < 0.0f) {
        const float swapped = u;
        u = 1.0f - v;
        v = 1.0f - swapped;
    }
    u = metal::copysign(u, unit.x);
    v = metal::copysign(v, unit.y);
    return encode_coordinate(u) | (encode_coordinate(v) << unit_y_shift);
}

inline float3 decode_unit(uint encoded) {
    const float u = decode_coordinate(encoded & unit_code_mask);
    const float v = decode_coordinate(encoded >> unit_y_shift);
    const float up = metal::abs(u);
    const float vp = metal::abs(v);
    const float signed_distance = 1.0f - (up + vp);
    const float r = 1.0f - metal::abs(signed_distance);
    const float phi = (r == 0.0f ? 1.0f : (vp - up) / r + 1.0f) * (M_PI_F / 4.0f);
    const float z = metal::copysign(1.0f - r * r, signed_distance);
    const float s = r * metal::sqrt(metal::max(0.0f, 2.0f - r * r));
    return metal::normalize(float3(metal::copysign(metal::cos(phi), u) * s, metal::copysign(metal::sin(phi), v) * s, z));
}

}  // namespace shaders
}  // namespace serenity
