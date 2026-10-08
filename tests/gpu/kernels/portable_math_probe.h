#pragma once

// The operations tests/gpu/portable_math_test.cpp compares between the CPU
// and the GPU. Compiled as C++ by the test and as Metal by
// portable_math_probe.metal: one source, so a difference in the results is a
// difference in the builds, never in the code.

#include "core/portable_math.h"

namespace serenity {
namespace probe {

enum Operation {
    op_add,
    op_sub,
    op_mul,
    op_div,
    op_sqrt,  // of a * a + b * b, so the operand itself is computed
    op_fma,
    op_mad,   // a * b + c as written: fused only if a compiler contracts it
    operation_count
};

inline float apply(int operation, float a, float b, float c) {
    switch (operation) {
    case op_add: return a + b;
    case op_sub: return a - b;
    case op_mul: return a * b;
    case op_div: return a / b;
    case op_sqrt: return portable::sqrt(a * a + b * b);
    case op_fma: return portable::fma(a, b, c);
    case op_mad: return a * b + c;
    default: return 0.0f;
    }
}

}  // namespace probe
}  // namespace serenity
