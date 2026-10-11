#pragma once

// The records bsdf_probe.metal and tests/gpu/bsdf_test.cpp share (Material:
// the BSDF's questions): one definition for both sides, fixed-width fields,
// counts and indices as integers, each size asserted on both sides and
// trivial copyability on the host (core/contracts/shared_layout.h gives the
// rules; SF.3, I.4). Included by that kernel and that test alone, so a change
// to one probe's records touches no other probe (SF.11;
// docs/architecture/change-axes.md). The kernel binds its output at buffer 0
// and its inputs after it in the order its comment lists them
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/shared_layout.h"

#include "core/contracts/float3.h"

namespace serenity {
namespace tests {

// Which wo a BSDF probe asks about, and how many draws (bsdf_samples) or
// cells a side (bsdf_density).
struct BsdfQuery {
    contracts::Float3 wo;
    uint32_t count;
};

static_assert(sizeof(BsdfQuery) == 16, "BsdfQuery must be the same 16 bytes on the host and in shaders");

// One draw of sample(), and what evaluate() and pdf() then say of its wi.
struct BsdfDraw {
    contracts::Float3 direction;  // the sample's wi
    float pdf;                    // the sample's pdf
    contracts::Float3 value;      // the sample's value
    uint32_t lobe;                // the sample's lobe bits
    contracts::Float3 evaluated;  // evaluate(wo, wi), asked afterwards
    float evaluated_pdf;          // pdf(wo, wi), asked afterwards
};

static_assert(sizeof(BsdfDraw) == 48, "BsdfDraw must be the same 48 bytes on the host and in shaders");

// evaluate() and pdf() at one cell of the density grid.
struct BsdfCell {
    contracts::Float3 evaluated;
    float pdf;
};

static_assert(sizeof(BsdfCell) == 16, "BsdfCell must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<BsdfQuery>, "BsdfQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<BsdfDraw>, "BsdfDraw is copied from the GPU as bytes");
static_assert(std::is_trivially_copyable_v<BsdfCell>, "BsdfCell is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
