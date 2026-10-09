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
// still.
//
// A path's numbers (the path tracer's, which averages frames) are drawn
// another way: independent, from a hash of the pixel, the frame's index and
// the dimension, the count of numbers the path drew before. Each frame's are
// independent of every other's, so their mean converges, and each is a
// function of where and when it is used, so any frame can be drawn again,
// alone (principle 2). The counter is the path's own, in a register, not
// state any other thread or frame shares.
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

// A path's numbers: `key` from the pixel and the frame, `dimension` counting.
struct PathNumbers {
    uint key;
    uint dimension;
};

inline PathNumbers path_numbers(uint2 pixel, uint frame) {
    return PathNumbers{pcg_hash(pixel.x ^ pcg_hash(pixel.y ^ pcg_hash(frame))), 0u};
}

// The path's next number, in [0, 1).
inline float next_number(thread PathNumbers& numbers) {
    return unit_float(pcg_hash(numbers.key ^ pcg_hash(numbers.dimension++)));
}

inline float2 next_numbers2(thread PathNumbers& numbers) {
    const float a = next_number(numbers);
    return float2(a, next_number(numbers));
}

inline float3 next_numbers3(thread PathNumbers& numbers) {
    const float a = next_number(numbers);
    const float b = next_number(numbers);
    return float3(a, b, next_number(numbers));
}

}  // namespace shaders
}  // namespace serenity
