// Runs Film's accumulation (metal/film/accumulate.metal.h) for
// tests/gpu/film_test.cpp: thread i folds samples[i] into pixels[i], starting
// over where starts[i] is 1, and every thread reports its failure to the
// shared counter, as the path pass does. Bindings: the output at 0, then the
// pixels, the samples, the starts, their count and the counter (probes.h).

#include <metal_stdlib>

#include "metal/film/accumulate.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::tests::Float4;

kernel void film_fold(device Float4* out [[buffer(0)]],
                      device const Float4* pixels [[buffer(1)]],
                      device const Float4* samples [[buffer(2)]],
                      device const uint* starts [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      device metal::atomic_uint* counter [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
    bool failed = false;
    if (i < count) {
        const float3 sample = float3(samples[i].x, samples[i].y, samples[i].z);
        failed = !finite(sample);
        const Float4 p = pixels[i];
        const float4 held = accumulate(float4(p.x, p.y, p.z, p.w), sample, starts[i] != 0u);
        out[i] = Float4{held.x, held.y, held.z, held.w};
    }
    count_non_finite(failed, counter);
}
