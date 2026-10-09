#pragma once

// Axis: Film (what a pixel accumulates).
//
// Folding a frame's radiance for a pixel into the mean of its samples so far:
// the accumulated image a converging pass keeps (metal/frame/accumulation.h
// holds it; the pass reads and writes it). Each pixel is four floats: the
// mean radiance in rgb, and in a the count of samples it is the mean of.
//
// The new pixel, n its count:
//
//   mean' = mean + (sample - mean) / (n + 1),   n' = n + 1
//
// the running mean in its stable form: it never forms mean x n, which can
// overflow where the mean itself cannot, and with n = 0 it is the sample. A
// pixel starting over (the frame holds no frames before it,
// accumulated_frames = 0) has n = 0 whatever the image held. The count is a
// float, exact to 2^24; the image holds at most 2^24 - 1 frames
// (accumulation.h), so n + 1 is always exact and never 0.
//
// A sample that is not finite (a NaN or an infinity in any channel) is a
// bug: a pdf of 0 divided by, a direction of no length. Folded in, it would
// poison the pixel for every frame after. It is left out: the pixel keeps
// its mean and its count, so the samples it does hold are weighed right
// (frame 0 failing and frame 1 radiance L gives L, not L / 2). And it is
// counted, in a counter Film keeps (metal/film/non_finite.h), which the
// headless renderer fails a run on and the window shows. It is not hidden,
// and it is not averaged in.
//
// The count is added to once per SIMD group, not once per sample: a
// systematic failure makes every pixel non-finite, and one atomic per pixel
// on one address would serialize millions of them in the frame that most
// needs reporting. count_non_finite() sums the group's failures with
// simd_sum and has one lane add them, so every thread of the group must call
// it, with false where it had no pixel.

#include <metal_atomic>
#include <metal_simdgroup>
#include <metal_stdlib>

namespace serenity {
namespace shaders {

inline bool finite(float3 sample) {
    return metal::all(metal::isfinite(sample));
}

// The pixel after folding in `sample`: `pixel` as the image held it, rgb the
// mean and a the count; `starting_over` when the frame holds no frames
// before it. A sample that is not finite leaves the pixel as it was (or
// empty, starting over); the caller reports it with count_non_finite().
inline float4 accumulate(float4 pixel, float3 sample, bool starting_over) {
    const float4 held = starting_over ? float4(0.0f) : pixel;
    if (!finite(sample)) {
        return held;
    }
    const float n = held.a;
    return float4(held.rgb + (sample - held.rgb) / (n + 1.0f), n + 1.0f);
}

// Adds to `counter` the number of threads in the SIMD group whose `failed`
// is true, with one atomic for the group. Every thread of the group calls it.
inline void count_non_finite(bool failed, device metal::atomic_uint* counter) {
    const uint failures = metal::simd_sum(failed ? 1u : 0u);
    if (failures != 0u && metal::simd_is_first()) {
        metal::atomic_fetch_add_explicit(counter, failures, metal::memory_order_relaxed);
    }
}

}  // namespace shaders
}  // namespace serenity
