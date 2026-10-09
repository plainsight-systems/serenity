#pragma once

// What tests/gpu/kernels/bsdf_probe.metal writes and tests/gpu/bsdf_test.cpp
// reads: one definition for both, fixed-width fields, the size asserted on
// each side (core/contracts/frame_constants.h gives the rules).

#if defined(__METAL_VERSION__)
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "core/contracts/float3.h"

namespace serenity {
namespace tests {

// One draw of sample(), and what evaluate() and pdf() then say of its wi.
struct Probe {
    contracts::Float3 direction;  // the sample's wi
    float pdf;                    // the sample's pdf
    contracts::Float3 value;      // the sample's value
    uint32_t lobe;                // the sample's lobe bits
    contracts::Float3 evaluated;  // evaluate(wo, wi), asked afterwards
    float evaluated_pdf;          // pdf(wo, wi), asked afterwards
};

static_assert(sizeof(Probe) == 48, "Probe must be the same 48 bytes on the host and in shaders");

}  // namespace tests
}  // namespace serenity
