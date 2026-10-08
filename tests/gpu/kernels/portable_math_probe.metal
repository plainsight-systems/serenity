// Applies every probe operation to each triple of inputs
// (kernels/portable_math_probe.h). Thread i reads in[3i .. 3i+2] and writes
// out[i * operation_count + op] for each operation.

#include <metal_stdlib>

#include "kernels/portable_math_probe.h"

kernel void portable_math_probe(device const float* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    const float a = in[3 * i];
    const float b = in[3 * i + 1];
    const float c = in[3 * i + 2];
    for (int op = 0; op < serenity::probe::operation_count; ++op) {
        out[i * serenity::probe::operation_count + op] = serenity::probe::apply(op, a, b, c);
    }
}
