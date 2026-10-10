#pragma once

// Shared shader mathematics: integer hashes, and bits to numbers in [0, 1).
// Mappings, owned by no family: the sampler draws its numbers from them
// (sampler/sampler.metal.h), the noise its lattice gradients
// (textures/noise.metal.h), the wood its boards (textures/wood.metal.h).
//
// The hashes are Jarzynski and Olano's ("Hash Functions for GPU Rendering",
// JCGT 2020): pcg_hash, PCG's output permutation of one 32-bit word, and
// pcg3d, their 3D hash of three. Each is a bijection of its input: every
// step (a multiply by an odd constant and an add per word, a word plus a
// product of the others, a word xor its own shift) can be undone, so
// different inputs never give the same output words (GDSA.3).

#include <metal_stdlib>

namespace serenity {
namespace shaders {

// PCG's (O'Neill 2014): the 32-bit LCG step, and the output permutation's
// multiplier.
constant constexpr uint pcg_multiplier = 747796405u;
constant constexpr uint pcg_increment = 2891336453u;
constant constexpr uint pcg_output_multiplier = 277803737u;

// pcg3d's first step, the LCG of Numerical Recipes, per word.
constant constexpr uint lcg_multiplier = 1664525u;
constant constexpr uint lcg_increment = 1013904223u;

// 2^-24: the spacing of floats in [0.5, 1), so the top 24 bits of a word
// times it is a float in [0, 1) exactly, never rounded up to 1.
constant constexpr float unit_spacing = 0x1p-24f;

inline uint pcg_hash(uint v) {
    const uint state = v * pcg_multiplier + pcg_increment;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * pcg_output_multiplier;
    return (word >> 22u) ^ word;
}

inline uint3 pcg3d(uint3 v) {
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
inline float unit_float(uint bits) {
    return float(bits >> 8u) * unit_spacing;
}

inline float3 unit_float3(uint3 bits) {
    return float3(bits >> 8u) * unit_spacing;
}

}  // namespace shaders
}  // namespace serenity
