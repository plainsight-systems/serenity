#pragma once

// Shared shader mathematics: an orthonormal frame about a direction, and
// warps from the unit square to the disk and the hemisphere. Mappings, not
// sources of numbers: the sampler (sampler/sampler.cuh) derives the
// numbers, and each family that turns numbers into a point or a direction
// (a material's sample(), a light's cone, the integrator's sky rays) maps
// them here. So no family reaches into another for a mapping, and the
// sampler can change its sequences without touching one. pbrt keeps the
// same split: its warps in util/sampling.h, apart from its Samplers.
//
// Translated line for line from Metal's metal/math/warp.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

// A point in the unit disk, area-preserving and continuous (Shirley and
// Chiu's concentric map), from a point in the unit square.
__device__ inline float2 concentric_disk(float2 u) {
  const float2 s = 2.0f * u - 1.0f;
  if (s.x == 0.0f && s.y == 0.0f) {
    return make_float2(0.0f);
  }
  const bool x_wider = abs(s.x) > abs(s.y);
  const float r = x_wider ? s.x : s.y;
  const float phi = x_wider ? (pi / 4.0f) * (s.y / s.x)
                            : (pi / 2.0f) - (pi / 4.0f) * (s.x / s.y);
  return r * make_float2(cos(phi), sin(phi));
}

// Two unit vectors that, with a unit normal n, make a right-handed
// orthonormal basis (t, b, n).
struct Tangents {
  float3 t;
  float3 b;
};

// The tangents of unit `n` (Duff et al., "Building an Orthonormal Basis,
// Revisited", 2017).
__device__ inline Tangents tangents(float3 n) {
  const float sign = copysign(1.0f, n.z);
  const float a = -1.0f / (sign + n.z);
  const float c = n.x * n.y * a;
  return Tangents{
      make_float3(1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x),
      make_float3(c, sign + n.y * n.y * a, -n.y)};
}

// A direction about unit `n`, distributed by the cosine to it (Malley's
// method), from a point in the unit square.
__device__ inline float3 cosine_direction(float3 n, float2 u) {
  const float2 d = concentric_disk(u);
  const Tangents f = tangents(n);
  return d.x * f.t + d.y * f.b + sqrt(max(0.0f, 1.0f - dot(d, d))) * n;
}

// The equal-area octahedral map between the unit sphere and the square
// (Clarberg 2008), quantized to two 16-bit numbers in one word, x in bits
// 0 to 15 and y in 16 to 31 (contracts/light_sample.h states it and what
// it loses): how a light's point on its unit sphere is stored (contract
// 4), and a surface's wo and its normals (contracts 5 and 2).
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
// the quantization is stated), measured by tests/gpu/encoding_test.cpp over
// the whole sphere.
//
// Cost: encode, a square root, an arctangent and a division; decode, a
// square root, a sine and a cosine. Pure (F.8), inline, no memory read.
inline constexpr float unit_levels = 65535.0f;  // 2^16 - 1, the largest code
// a coordinate's code, x's in the low bits
inline constexpr uint unit_code_mask = 0xffffu;
inline constexpr uint unit_y_shift = 16u;  // y's above it

// A coordinate in [-1, 1] as its 16-bit code, and back.
__device__ inline uint encode_coordinate(float c) {
  return uint(rint(clamp((c + 1.0f) * 0.5f, 0.0f, 1.0f) * unit_levels));
}

__device__ inline float decode_coordinate(uint code) {
  return float(code) * (2.0f / unit_levels) - 1.0f;
}

__device__ inline uint encode_unit(float3 unit) {
  const float3 a = abs(unit);
  const float r = sqrt((a.x * a.x + a.y * a.y) / (1.0f + a.z));
  const float wide = max(a.x, a.y);
  const float narrow = min(a.x, a.y);
  float phi = wide > 0.0f ? atan(narrow / wide) * (2.0f / pi) : 0.0f;
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
  u = copysign(u, unit.x);
  v = copysign(v, unit.y);
  return encode_coordinate(u) | (encode_coordinate(v) << unit_y_shift);
}

__device__ inline float3 decode_unit(uint encoded) {
  const float u = decode_coordinate(encoded & unit_code_mask);
  const float v = decode_coordinate(encoded >> unit_y_shift);
  const float up = abs(u);
  const float vp = abs(v);
  const float signed_distance = 1.0f - (up + vp);
  const float r = 1.0f - abs(signed_distance);
  const float phi = (r == 0.0f ? 1.0f : (vp - up) / r + 1.0f) * (pi / 4.0f);
  const float z = copysign(1.0f - r * r, signed_distance);
  const float s = r * sqrt(max(0.0f, 2.0f - r * r));
  return normalize(
      make_float3(copysign(cos(phi), u) * s, copysign(sin(phi), v) * s, z));
}

}  // namespace shaders
}  // namespace serenity
