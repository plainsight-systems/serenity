#pragma once

// The records film_probe.metal and tests/gpu/film_test.cpp share (the
// accumulated image's pixels): one definition for both sides, fixed-width
// fields, counts and indices as integers, each size asserted on both sides
// and trivial copyability on the host (core/contracts/shared_layout.h gives
// the rules; SF.3, I.4). Included by that kernel and that test alone, so a
// change to one probe's records touches no other probe (SF.11;
// docs/architecture/change-axes.md). The kernel binds its output at buffer 0
// and its inputs after it in the order its comment lists them
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// Four floats, a shading language float4's sixteen bytes.
struct Float4 {
    float x;
    float y;
    float z;
    float w;
};

static_assert(sizeof(Float4) == 16, "Float4 must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<Float4>, "Float4 is copied to and from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
