#pragma once

// Axis: GPU backend (device code).
//
// The vector arithmetic Metal's shading language has and CUDA's does not
// (docs/research/2026-10-19-cuda-backend.md, "What carries over": its
// float2, float3 and float4 are structs of floats, with no operators and
// no functions over them): the operators, and the functions the shaders
// call, under the shading language's names (dot, normalize, mix, saturate,
// select, ...), so a translated header reads line for line against its
// source in src/metal/. Each is Metal's definition (the Metal Shading
// Language Specification's common, math and geometric functions), computed
// in IEEE single precision: the backend compiles
// without fast math (cmake/CudaModules.cmake), so division and square root
// are correctly rounded where Metal's fast math approximates them.
//
// Constructors are make_float3() and its kin, CUDA's names, with the
// shading language's other forms beside them: a splat (make_float3(s)),
// and a vector from a shorter one and a component. Swizzles are functions:
// xy(v), yz(v), xz(v), xyz(v), and rgb(v) for a colour. A comparison of
// vectors is a bool3, which any() and all() reduce, as Metal's.
//
// Everything is declared in serenity::shaders, beside the translated
// headers, which call it unqualified; the scalar forms are here too, since
// a name declared in the namespace hides CUDA's global one (min, max, abs,
// sqrt) for every unqualified call in it. __device__ functions, inline
// (F.5: small, called in every inner loop), constexpr where CUDA allows.

#include <cuda/std/cstdint>
#include <cuda/std/limits>

namespace serenity {
namespace shaders {

using uint = unsigned int;
using ushort = unsigned short;

// The shading language's M_PI_F and M_1_PI_F, as constants rather than
// macros (ES.31).
inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float inverse_pi = 0.318309886183790671538f;

// --- Constructors -----------------------------------------------------------

// CUDA's own, of every component, named here so the forms below add to
// them rather than hide them (a declaration in the namespace hides the
// global one).
using ::make_float2;
using ::make_float3;
using ::make_float4;
using ::make_int2;
using ::make_int3;
using ::make_uint2;
using ::make_uint3;

__device__ inline float2 make_float2(float s) { return float2{s, s}; }
__device__ inline float2 make_float2(uint2 v) {
  return float2{static_cast<float>(v.x), static_cast<float>(v.y)};
}
__device__ inline float3 make_float3(float s) { return float3{s, s, s}; }
__device__ inline float3 make_float3(float2 v, float z) {
  return float3{v.x, v.y, z};
}
__device__ inline float3 make_float3(uint3 v) {
  return float3{static_cast<float>(v.x), static_cast<float>(v.y),
                static_cast<float>(v.z)};
}
__device__ inline float3 make_float3(int3 v) {
  return float3{static_cast<float>(v.x), static_cast<float>(v.y),
                static_cast<float>(v.z)};
}
__device__ inline float4 make_float4(float s) { return float4{s, s, s, s}; }
__device__ inline float4 make_float4(float3 v, float w) {
  return float4{v.x, v.y, v.z, w};
}
// Toward zero, as the shading language's int3(float3).
__device__ inline int3 make_int3(float3 v) {
  return int3{static_cast<int>(v.x), static_cast<int>(v.y),
              static_cast<int>(v.z)};
}
// Toward zero, as the shading language's uint2(float2), for values the
// caller has made non-negative.
__device__ inline uint2 make_uint2(float2 v) {
  return uint2{static_cast<uint>(v.x), static_cast<uint>(v.y)};
}
__device__ inline uint2 make_uint2(uint s) { return uint2{s, s}; }
// The shading language's uint2(int2), for values the caller has made
// non-negative, and int2(uint2).
__device__ inline uint2 make_uint2(int2 v) {
  return uint2{static_cast<uint>(v.x), static_cast<uint>(v.y)};
}
__device__ inline int2 make_int2(uint2 v) {
  return int2{static_cast<int>(v.x), static_cast<int>(v.y)};
}
__device__ inline int2 make_int2(int s) { return int2{s, s}; }
__device__ inline uint3 make_uint3(uint s) { return uint3{s, s, s}; }
__device__ inline uint3 make_uint3(uint2 v, uint z) {
  return uint3{v.x, v.y, z};
}

// --- Swizzles ---------------------------------------------------------------

__device__ inline int2 yx(int2 v) { return int2{v.y, v.x}; }
__device__ inline float2 xy(float3 v) { return float2{v.x, v.y}; }
__device__ inline float2 yz(float3 v) { return float2{v.y, v.z}; }
__device__ inline float2 xz(float3 v) { return float2{v.x, v.z}; }
__device__ inline float2 yz(float4 v) { return float2{v.y, v.z}; }
__device__ inline float3 xyz(float4 v) { return float3{v.x, v.y, v.z}; }
__device__ inline float3 rgb(float4 v) { return float3{v.x, v.y, v.z}; }

// --- Bitwise reinterpretation (the shading language's as_type) --------------

__device__ inline uint as_uint(float v) { return __float_as_uint(v); }
__device__ inline float as_float(uint v) { return __uint_as_float(v); }
__device__ inline uint3 as_uint3(float3 v) {
  return uint3{__float_as_uint(v.x), __float_as_uint(v.y),
               __float_as_uint(v.z)};
}
__device__ inline float3 as_float3(uint3 v) {
  return float3{__uint_as_float(v.x), __uint_as_float(v.y),
                __uint_as_float(v.z)};
}
__device__ inline uint as_uint(int v) { return static_cast<uint>(v); }

// --- Components by index (the shading language's v[i])

__device__ inline float component(float3 v, uint i) {
  return i == 0u ? v.x : (i == 1u ? v.y : v.z);
}
__device__ inline float component(float4 v, uint i) {
  return i == 0u ? v.x : (i == 1u ? v.y : (i == 2u ? v.z : v.w));
}

// --- Comparisons
// --------------------------------------------------------------

struct bool3 {
  bool x;
  bool y;
  bool z;
};

struct bool2 {
  bool x;
  bool y;
};

__device__ inline bool any(bool2 b) { return b.x || b.y; }
__device__ inline bool all(bool2 b) { return b.x && b.y; }
__device__ inline bool2 operator>=(float2 a, float s) {
  return bool2{a.x >= s, a.y >= s};
}
__device__ inline bool2 operator<(float2 a, float2 b) {
  return bool2{a.x < b.x, a.y < b.y};
}
__device__ inline bool2 operator<(int2 a, int2 b) {
  return bool2{a.x < b.x, a.y < b.y};
}
__device__ inline bool2 operator>=(int2 a, int2 b) {
  return bool2{a.x >= b.x, a.y >= b.y};
}
__device__ inline bool any(bool3 b) { return b.x || b.y || b.z; }
__device__ inline bool all(bool3 b) { return b.x && b.y && b.z; }
__device__ inline bool3 operator<(float3 a, float s) {
  return bool3{a.x < s, a.y < s, a.z < s};
}
__device__ inline bool3 operator<=(float3 a, float s) {
  return bool3{a.x <= s, a.y <= s, a.z <= s};
}
__device__ inline bool3 operator>(float3 a, float s) {
  return bool3{a.x > s, a.y > s, a.z > s};
}
__device__ inline bool3 operator!=(float3 a, float s) {
  return bool3{a.x != s, a.y != s, a.z != s};
}
__device__ inline bool3 operator>(float3 a, float3 b) {
  return bool3{a.x > b.x, a.y > b.y, a.z > b.z};
}
__device__ inline bool3 operator<(float3 a, float3 b) {
  return bool3{a.x < b.x, a.y < b.y, a.z < b.z};
}

// --- Arithmetic: float2
// -------------------------------------------------------

__device__ inline float2 operator+(float2 a, float2 b) {
  return float2{a.x + b.x, a.y + b.y};
}
__device__ inline float2 operator-(float2 a, float2 b) {
  return float2{a.x - b.x, a.y - b.y};
}
__device__ inline float2 operator-(float2 a) { return float2{-a.x, -a.y}; }
__device__ inline float2 operator*(float2 a, float2 b) {
  return float2{a.x * b.x, a.y * b.y};
}
__device__ inline float2 operator*(float2 a, float s) {
  return float2{a.x * s, a.y * s};
}
__device__ inline float2 operator*(float s, float2 a) {
  return float2{s * a.x, s * a.y};
}
__device__ inline float2 operator/(float2 a, float2 b) {
  return float2{a.x / b.x, a.y / b.y};
}
__device__ inline float2 operator/(float2 a, float s) {
  return float2{a.x / s, a.y / s};
}
__device__ inline float2 operator/(float s, float2 a) {
  return float2{s / a.x, s / a.y};
}
__device__ inline float2 operator+(float2 a, float s) {
  return float2{a.x + s, a.y + s};
}
__device__ inline float2 operator-(float2 a, float s) {
  return float2{a.x - s, a.y - s};
}
__device__ inline float2 operator-(float s, float2 a) {
  return float2{s - a.x, s - a.y};
}
__device__ inline float2& operator+=(float2& a, float2 b) { return a = a + b; }
__device__ inline float2& operator*=(float2& a, float s) { return a = a * s; }

// --- Arithmetic: float3
// -------------------------------------------------------

__device__ inline float3 operator+(float3 a, float3 b) {
  return float3{a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ inline float3 operator-(float3 a, float3 b) {
  return float3{a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ inline float3 operator-(float3 a) {
  return float3{-a.x, -a.y, -a.z};
}
__device__ inline float3 operator*(float3 a, float3 b) {
  return float3{a.x * b.x, a.y * b.y, a.z * b.z};
}
__device__ inline float3 operator*(float3 a, float s) {
  return float3{a.x * s, a.y * s, a.z * s};
}
__device__ inline float3 operator*(float s, float3 a) {
  return float3{s * a.x, s * a.y, s * a.z};
}
__device__ inline float3 operator/(float3 a, float3 b) {
  return float3{a.x / b.x, a.y / b.y, a.z / b.z};
}
__device__ inline float3 operator/(float3 a, float s) {
  return float3{a.x / s, a.y / s, a.z / s};
}
__device__ inline float3 operator/(float s, float3 a) {
  return float3{s / a.x, s / a.y, s / a.z};
}
__device__ inline float3 operator+(float3 a, float s) {
  return float3{a.x + s, a.y + s, a.z + s};
}
__device__ inline float3 operator+(float s, float3 a) {
  return float3{s + a.x, s + a.y, s + a.z};
}
__device__ inline float3 operator-(float3 a, float s) {
  return float3{a.x - s, a.y - s, a.z - s};
}
__device__ inline float3 operator-(float s, float3 a) {
  return float3{s - a.x, s - a.y, s - a.z};
}
__device__ inline float3& operator+=(float3& a, float3 b) { return a = a + b; }
__device__ inline float3& operator-=(float3& a, float3 b) { return a = a - b; }
__device__ inline float3& operator-=(float3& a, float s) { return a = a - s; }
__device__ inline float3& operator*=(float3& a, float3 b) { return a = a * b; }
__device__ inline float3& operator*=(float3& a, float s) { return a = a * s; }
__device__ inline float3& operator/=(float3& a, float s) { return a = a / s; }

// --- Arithmetic: float4
// -------------------------------------------------------

__device__ inline float4 operator+(float4 a, float4 b) {
  return float4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}
__device__ inline float4 operator-(float4 a, float4 b) {
  return float4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}
__device__ inline float4 operator-(float4 a) {
  return float4{-a.x, -a.y, -a.z, -a.w};
}
__device__ inline float4 operator*(float4 a, float4 b) {
  return float4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}
__device__ inline float4 operator*(float4 a, float s) {
  return float4{a.x * s, a.y * s, a.z * s, a.w * s};
}
__device__ inline float4 operator*(float s, float4 a) {
  return float4{s * a.x, s * a.y, s * a.z, s * a.w};
}
__device__ inline float4 operator/(float4 a, float s) {
  return float4{a.x / s, a.y / s, a.z / s, a.w / s};
}
__device__ inline float4& operator+=(float4& a, float4 b) { return a = a + b; }

// --- Arithmetic: integer vectors

__device__ inline int2 operator+(int2 a, int2 b) {
  return int2{a.x + b.x, a.y + b.y};
}
__device__ inline int2 operator-(int2 a, int2 b) {
  return int2{a.x - b.x, a.y - b.y};
}
__device__ inline int2 operator*(int s, int2 a) {
  return int2{s * a.x, s * a.y};
}
// C++'s remainder, the sign of the dividend's, as the shading language's.
__device__ inline int2 operator%(int2 a, int s) {
  return int2{a.x % s, a.y % s};
}
__device__ inline int2 operator+(int2 a, int s) {
  return int2{a.x + s, a.y + s};
}
__device__ inline uint2 operator/(uint2 a, uint s) {
  return uint2{a.x / s, a.y / s};
}
__device__ inline uint2 operator-(uint2 a, uint s) {
  return uint2{a.x - s, a.y - s};
}
__device__ inline uint2 min(uint2 a, uint2 b) {
  return uint2{a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y};
}
// ----------------------------------------------

__device__ inline int3 operator+(int3 a, int3 b) {
  return int3{a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ inline int3 operator-(int3 a) { return int3{-a.x, -a.y, -a.z}; }
__device__ inline uint3 operator+(uint3 a, uint3 b) {
  return uint3{a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ inline uint3 operator*(uint3 a, uint3 b) {
  return uint3{a.x * b.x, a.y * b.y, a.z * b.z};
}
__device__ inline uint3 operator*(uint3 a, uint s) {
  return uint3{a.x * s, a.y * s, a.z * s};
}
__device__ inline uint3 operator+(uint3 a, uint s) {
  return uint3{a.x + s, a.y + s, a.z + s};
}
__device__ inline uint3 operator^(uint3 a, uint3 b) {
  return uint3{a.x ^ b.x, a.y ^ b.y, a.z ^ b.z};
}
__device__ inline uint3 operator>>(uint3 a, uint s) {
  return uint3{a.x >> s, a.y >> s, a.z >> s};
}
__device__ inline uint3& operator+=(uint3& a, uint3 b) { return a = a + b; }
__device__ inline uint3& operator^=(uint3& a, uint3 b) { return a = a ^ b; }
// A signed vector added to the bits of an unsigned one, as the shading
// language's as_type<uint3>(int3): two's complement, modulo 2^32.
__device__ inline uint3 as_uint3(int3 v) {
  return uint3{static_cast<uint>(v.x), static_cast<uint>(v.y),
               static_cast<uint>(v.z)};
}

// --- Common functions, scalar
// --------------------------------------------------

__device__ inline float min(float a, float b) { return fminf(a, b); }
__device__ inline float max(float a, float b) { return fmaxf(a, b); }
__device__ inline uint min(uint a, uint b) { return a < b ? a : b; }
__device__ inline uint max(uint a, uint b) { return a < b ? b : a; }
__device__ inline int min(int a, int b) { return a < b ? a : b; }
__device__ inline int max(int a, int b) { return a < b ? b : a; }
__device__ inline float abs(float a) { return fabsf(a); }
__device__ inline float sqrt(float a) { return sqrtf(a); }
__device__ inline float rsqrt(float a) { return rsqrtf(a); }
__device__ inline float floor(float a) { return floorf(a); }
__device__ inline float rint(float a) { return rintf(a); }
__device__ inline float exp(float a) { return expf(a); }
__device__ inline float exp2(float a) { return exp2f(a); }
__device__ inline float pow(float a, float b) { return powf(a, b); }
__device__ inline float sin(float a) { return sinf(a); }
__device__ inline float cos(float a) { return cosf(a); }
__device__ inline float atan(float a) { return atanf(a); }
__device__ inline float atan2(float y, float x) { return atan2f(y, x); }
__device__ inline float fmod(float a, float b) { return fmodf(a, b); }
__device__ inline float copysign(float a, float b) { return copysignf(a, b); }
__device__ inline bool isfinite(float a) { return ::isfinite(a); }
__device__ inline float clamp(float x, float lo, float hi) {
  return min(max(x, lo), hi);
}
__device__ inline uint clamp(uint x, uint lo, uint hi) {
  return min(max(x, lo), hi);
}
__device__ inline float saturate(float x) { return clamp(x, 0.0f, 1.0f); }
__device__ inline float mix(float x, float y, float a) {
  return x + (y - x) * a;
}
// 1 above 0, -1 below, 0 for either zero (keeping its sign) and for NaN.
__device__ inline float sign(float x) {
  return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : (x == x ? x : 0.0f));
}
// x - floor(x), at most the float below 1.
__device__ inline float fract(float x) {
  return fminf(x - floorf(x), 0x1.fffffep-1f);
}
__device__ inline float step(float edge, float x) {
  return x < edge ? 0.0f : 1.0f;
}
__device__ inline float smoothstep(float edge0, float edge1, float x) {
  const float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}
__device__ inline float select(float a, float b, bool c) { return c ? b : a; }

// --- Common functions, vectors
// -------------------------------------------------

__device__ inline float3 min(float3 a, float3 b) {
  return float3{fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}
__device__ inline float3 max(float3 a, float3 b) {
  return float3{fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}
__device__ inline float3 min(float3 a, float s) {
  return min(a, make_float3(s));
}
__device__ inline float3 max(float3 a, float s) {
  return max(a, make_float3(s));
}
__device__ inline float3 abs(float3 a) {
  return float3{fabsf(a.x), fabsf(a.y), fabsf(a.z)};
}
__device__ inline float3 sqrt(float3 a) {
  return float3{sqrtf(a.x), sqrtf(a.y), sqrtf(a.z)};
}
__device__ inline float3 exp(float3 a) {
  return float3{expf(a.x), expf(a.y), expf(a.z)};
}
__device__ inline float3 floor(float3 a) {
  return float3{floorf(a.x), floorf(a.y), floorf(a.z)};
}
__device__ inline float2 floor(float2 a) {
  return float2{floorf(a.x), floorf(a.y)};
}
__device__ inline float2 rint(float2 a) {
  return float2{rintf(a.x), rintf(a.y)};
}
__device__ inline float3 fract(float3 a) {
  return float3{fract(a.x), fract(a.y), fract(a.z)};
}
__device__ inline float3 sign(float3 a) {
  return float3{sign(a.x), sign(a.y), sign(a.z)};
}
__device__ inline float3 pow(float3 a, float b) {
  return float3{powf(a.x, b), powf(a.y, b), powf(a.z, b)};
}
__device__ inline float3 pow(float3 a, float3 b) {
  return float3{powf(a.x, b.x), powf(a.y, b.y), powf(a.z, b.z)};
}
__device__ inline float3 clamp(float3 x, float lo, float hi) {
  return float3{clamp(x.x, lo, hi), clamp(x.y, lo, hi), clamp(x.z, lo, hi)};
}
__device__ inline float3 saturate(float3 x) { return clamp(x, 0.0f, 1.0f); }
__device__ inline float3 mix(float3 x, float3 y, float a) {
  return x + (y - x) * a;
}
__device__ inline float3 mix(float3 x, float3 y, float3 a) {
  return x + (y - x) * a;
}
__device__ inline float3 smoothstep(float edge0, float edge1, float3 x) {
  return float3{smoothstep(edge0, edge1, x.x), smoothstep(edge0, edge1, x.y),
                smoothstep(edge0, edge1, x.z)};
}
__device__ inline float3 step(float3 edge, float3 x) {
  return float3{step(edge.x, x.x), step(edge.y, x.y), step(edge.z, x.z)};
}
__device__ inline float3 select(float3 a, float3 b, bool3 c) {
  return float3{c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z};
}
__device__ inline float3 select(float3 a, float3 b, bool c) {
  return c ? b : a;
}
__device__ inline int3 select(int3 a, int3 b, bool3 c) {
  return int3{c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z};
}
__device__ inline bool3 isfinite(float3 a) {
  return bool3{::isfinite(a.x), ::isfinite(a.y), ::isfinite(a.z)};
}

// --- Geometric functions
// -------------------------------------------------------

__device__ inline float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}
__device__ inline float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline float3 cross(float3 a, float3 b) {
  return float3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x};
}
__device__ inline float length(float2 a) { return sqrtf(dot(a, a)); }
__device__ inline float length(float3 a) { return sqrtf(dot(a, a)); }
__device__ inline float distance(float3 a, float3 b) { return length(a - b); }
__device__ inline float3 normalize(float3 a) { return a / length(a); }
__device__ inline float3 reflect(float3 i, float3 n) {
  return i - 2.0f * dot(n, i) * n;
}

}  // namespace shaders
}  // namespace serenity
