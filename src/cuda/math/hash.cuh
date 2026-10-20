#pragma once

// Shared shader mathematics: integer hashes, and bits to numbers in [0, 1).
// Mappings, owned by no family: the sampler draws its numbers from them
// (sampler/sampler.cuh), the noise its lattice gradients
// (textures/noise.cuh), the wood its boards (textures/wood.cuh).
//
// The hashes are Jarzynski and Olano's ("Hash Functions for GPU Rendering",
// JCGT 2020): pcg_hash, PCG's output permutation of one 32-bit word, and
// pcg3d, their 3D hash of three. Each is a bijection of its input: every
// step (a multiply by an odd constant and an add per word, a word plus a
// product of the others, a word xor its own shift) can be undone, so
// different inputs never give the same output words (GDSA.3).
//
// Translated line for line from Metal's metal/math/hash.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

// 2^32 / phi, its integer part (odd): a multiplier that spreads consecutive
// integers across the word before they are hashed together (Knuth's
// multiplicative hashing).
inline constexpr uint golden_ratio_32 = 0x9e3779b9u;

// PCG's (O'Neill 2014): the 32-bit LCG step, and the output permutation's
// multiplier.
inline constexpr uint pcg_multiplier = 747796405u;
inline constexpr uint pcg_increment = 2891336453u;
inline constexpr uint pcg_output_multiplier = 277803737u;

// pcg3d's first step, the LCG of Numerical Recipes, per word.
inline constexpr uint lcg_multiplier = 1664525u;
inline constexpr uint lcg_increment = 1013904223u;

// 2^-24: the spacing of floats in [0.5, 1), so the top 24 bits of a word
// times it is a float in [0, 1) exactly, never rounded up to 1.
inline constexpr float unit_spacing = 0x1p-24f;

__device__ inline uint pcg_hash(uint v) {
  const uint state = v * pcg_multiplier + pcg_increment;
  const uint word =
      ((state >> ((state >> 28u) + 4u)) ^ state) * pcg_output_multiplier;
  return (word >> 22u) ^ word;
}

__device__ inline uint3 pcg3d(uint3 v) {
  v = v * lcg_multiplier + lcg_increment;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  v ^= v >> 16u;
  v.x += v.y * v.z;
  v.y += v.z * v.x;
  v.z += v.x * v.y;
  return v;
}

// A number in [0, 1) from the top 24 bits of `bits`.
__device__ inline float unit_float(uint bits) {
  return float(bits >> 8u) * unit_spacing;
}

__device__ inline float3 unit_float3(uint3 bits) {
  return make_float3(bits >> 8u) * unit_spacing;
}

}  // namespace shaders
}  // namespace serenity
