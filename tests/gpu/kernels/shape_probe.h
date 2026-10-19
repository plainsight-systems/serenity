#pragma once

// The records shape_probe.metal and tests/gpu/shape_test.cpp share (Shape: a
// surface's interaction): one definition for both sides, fixed-width fields,
// counts and indices as integers, each size asserted on both sides and
// trivial copyability on the host (core/contracts/shared_layout.h gives the
// rules; SF.3, I.4). Included by that kernel and that test alone, so a change
// to one probe's records touches no other probe (SF.11;
// docs/architecture/change-axes.md). The kernel binds its output at buffer 0
// and its inputs after it in the order its comment lists them
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// A point on shape `shape`, where a ray arriving from +z meets it.
struct ShapeQuery {
  contracts::Float3 point;
  uint32_t shape;
};

static_assert(
    sizeof(ShapeQuery) == 16,
    "ShapeQuery must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<ShapeQuery>,
              "ShapeQuery is copied to the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
