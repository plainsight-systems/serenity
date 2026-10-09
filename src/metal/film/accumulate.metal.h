#pragma once

// Axis: Film (what a pixel accumulates).
//
// Folding a frame's radiance for a pixel into the mean of the frames before
// it: the accumulated image a converging pass keeps (metal/frame/
// accumulation.h holds it; the pass reads and writes it).
//
// The new mean, n the frames the image already held:
//
//   mean' = mean + (sample - mean) / (n + 1)
//
// the running mean in its stable form: it never forms mean x n, which can
// overflow where the mean itself cannot, and with n = 0 it is the sample.
//
// A sample that is not finite (a NaN or an infinity in any channel) is a
// bug: a pdf of 0 divided by, a direction of no length. Folded in, it would
// poison the pixel for every frame after. It is left out, the mean kept as
// it was (0 for an image starting over, whose old contents are not a mean of
// anything), and counted, once per sample, in a counter the accumulated
// image keeps (Accumulation::non_finite_samples()), which the headless
// renderer fails a run on and the window shows. It is not hidden, and it is
// not averaged in.

#include <metal_atomic>
#include <metal_stdlib>

namespace serenity {
namespace shaders {

// The mean after folding in `sample`, or the mean unchanged, `non_finite`
// counting the sample, if it is not finite.
inline float3 accumulate(float3 mean, float3 sample, uint held, device metal::atomic_uint* non_finite) {
    if (!metal::all(metal::isfinite(sample))) {
        metal::atomic_fetch_add_explicit(non_finite, 1u, metal::memory_order_relaxed);
        return held == 0u ? float3(0.0f) : mean;
    }
    return mean + (sample - mean) / float(held + 1u);
}

}  // namespace shaders
}  // namespace serenity
