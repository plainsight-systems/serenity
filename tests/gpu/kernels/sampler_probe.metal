// Runs the numbers a path draws (metal/sampler/sampler.metal.h) for
// tests/gpu/sampler_test.cpp. Bindings: the output at 0, the inputs after
// it in the order the kernel lists them (probes.h).

#include <metal_stdlib>

#include "metal/sampler/sampler.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::tests::PathNumbersDraw;
using serenity::tests::PathNumbersQuery;

// The first path_numbers_drawn numbers of the path through pixel
// (queries[i].x, queries[i].y) in frame queries[i].frame. Inputs: the
// queries, their count.
kernel void path_numbers_probe(device PathNumbersDraw* out [[buffer(0)]],
                               device const PathNumbersQuery* queries [[buffer(1)]],
                               constant uint& count [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    PathNumbers numbers = path_numbers(uint2(queries[i].x, queries[i].y), queries[i].frame);
    PathNumbersDraw drawn;
    for (uint k = 0; k < serenity::tests::path_numbers_drawn; ++k) {
        drawn.u[k] = next_number(numbers);
    }
    out[i] = drawn;
}
