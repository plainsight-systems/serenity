#pragma once

// The records leave_probe.metal and tests/gpu/leave_test.cpp share
// (Integrator: a ray leaving a surface, integrator/surface.metal.h): one
// definition for both sides, fixed-width fields, counts and indices as
// integers, each size asserted on both sides and trivial copyability on the
// host (core/contracts/shared_layout.h gives the rules; SF.3, I.4). Included
// by that kernel and that test alone, so a change to one probe's records
// touches no other probe (SF.11; docs/architecture/change-axes.md). The
// kernel binds its output at buffer 0 and its inputs after it in the order
// its comment lists them.

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// How many rays leave each surface point the probe finds: one each way, out
// along the geometric normal's side and in along the other, at each of
// leave_cosines' angles to it.
SERENITY_CONSTANT uint32_t leave_cosine_count = 8u;
SERENITY_CONSTANT uint32_t leave_directions = 2u * leave_cosine_count;

// A ray from `origin` along unit `direction`, aimed at shape `shape`: the
// path tracer's step 1, to find a surface point the way it does.
struct LeaveQuery {
  contracts::Float3 origin;
  uint32_t shape;
  contracts::Float3 direction;
  uint32_t padding;
};

static_assert(
    sizeof(LeaveQuery) == 32,
    "LeaveQuery must be the same 32 bytes on the host and in shaders");

// One ray leaving the point a query found: whether the query found its
// shape first; the side the leaving ray leaves on, as leave() decides it;
// its cosine to the geometric normal; and what it hit, its distance from
// where it starts.
struct LeaveResult {
  uint32_t found;    // the query's ray hit its shape, nearest
  uint32_t outward;  // leaves_outward(): on the geometric normal's side
  uint32_t hit;      // the leaving ray hit a shape
  uint32_t hit_shape;
  float t;       // the leaving ray's hit distance, where it hit
  float cosine;  // its direction's dot product with the geometric normal
  uint32_t padding[2];
};

static_assert(
    sizeof(LeaveResult) == 32,
    "LeaveResult must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<LeaveQuery>,
              "LeaveQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<LeaveResult>,
              "LeaveResult is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
