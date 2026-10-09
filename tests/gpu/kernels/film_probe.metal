// Runs Film's accumulation (metal/film/accumulate.metal.h) for
// tests/gpu/film_test.cpp: thread i folds samples[i] into pixels[i], starting
// over where starts[i] is 1, and every thread reports its failure to the
// shared counter, as the path pass does.

#include <metal_stdlib>

#include "metal/film/accumulate.metal.h"

using namespace serenity::shaders;

kernel void film_fold(device float4* out [[buffer(0)]],
                      device const float4* pixels [[buffer(1)]],
                      device const float4* samples [[buffer(2)]],
                      device const uint* starts [[buffer(3)]],
                      constant uint& count [[buffer(4)]],
                      device metal::atomic_uint* counter [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
    bool failed = false;
    if (i < count) {
        const float3 sample = samples[i].xyz;
        failed = !finite(sample);
        out[i] = accumulate(pixels[i], sample, starts[i] != 0u);
    }
    count_non_finite(failed, counter);
}
