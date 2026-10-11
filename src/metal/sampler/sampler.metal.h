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
// another way: independent, each the first word of pcg3d (math/
// hash.metal.h) of three, the pixel's key, the frame's index and the
// dimension, the count of numbers the path drew before. Counter-based, as
// GDSA.3 asks: the stream is (pixel, frame), 64 bits, and pcg3d is a
// bijection of its three words, so no two pixels or frames share a stream;
// two numbers agree only as two independent 32-bit words do, by chance.
// (A 32-bit hash of pixel and frame as the key, as before, gave some 6.9e3
// pairs of pixels per frame at 3456 x 2234 the same key, and so the same
// numbers all the way down their paths.) The pixel's key is x + 2^16 y,
// distinct for every pixel of an image up to 65536 wide, past Metal's
// largest texture. Each frame's numbers are independent of every other's,
// so their mean converges, and each is a function of where and when it is
// used, so any frame can be drawn again, alone (principle 2). The counter is
// the path's own, in a register, not state any other thread or frame shares.

#include <metal_stdlib>

#include "metal/math/hash.metal.h"

namespace serenity {
namespace shaders {

// R2's step: 1 / p and 1 / p^2, p the plastic constant, to a float's
// precision.
constant constexpr float2 r2_step = float2(0.754877666f, 0.569840291f);

// The offset of `pixel`'s samples for `purpose`.
inline float2 sample_offset(uint2 pixel, uint purpose) {
    const uint h = pcg_hash(pixel.x ^ pcg_hash(pixel.y ^ pcg_hash(purpose)));
    return float2(unit_float(h), unit_float(pcg_hash(h)));
}

// Sample `index` of a pixel's set whose offset is `offset`.
inline float2 sample_2d(float2 offset, uint index) {
    return metal::fract(offset + float(index) * r2_step);
}

// A path's numbers: its stream, `key`, the pixel's key and the frame's
// index; `dimension` counting the numbers drawn.
struct PathNumbers {
    uint2 key;
    uint dimension;
};

inline PathNumbers path_numbers(uint2 pixel, uint frame) {
    return PathNumbers{uint2(pixel.x | (pixel.y << 16u), frame), 0u};
}

// The path's next number, in [0, 1).
inline float next_number(thread PathNumbers& numbers) {
    return unit_float(pcg3d(uint3(numbers.key, numbers.dimension++)).x);
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

// Another purpose's numbers in the same pixel and frame: `stream`, a
// path's (path_numbers()), from dimension `first`. ReSTIR DI's candidates,
// temporal and spatial passes draw from first = 1, 2 and 3 x 2^24, and its
// temporal permutation from 4 x 2^24 at pixel (0, 0), the frame's alone
// (metal/passes/restir_di/restir_di.metal.h), where no path reaches: the
// safety stop bounds a path at 256 surfaces of at most 7 numbers each, 4 for
// the camera, under 2^11 (path.metal.h). pcg3d is a bijection of (key,
// dimension), so the purposes' numbers are distinct words from the path's and
// from each other's, as independent as any two of the path's (GDSA.3: a
// stream per pixel and frame, a counter within it). The shade continues the
// path's own stream where the initial pass left it
// (contracts/surface_record.h, PathState::dimension). The stream as one
// value, not its pixel and frame, so a frame's index and a dimension, both
// words, cannot be passed in each other's places (I.24).
inline PathNumbers purpose_numbers(PathNumbers stream, uint first) {
    return PathNumbers{stream.key, first};
}

}  // namespace shaders
}  // namespace serenity
