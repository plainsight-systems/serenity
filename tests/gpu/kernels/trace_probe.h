#pragma once

// The records trace_probe.metal and tests/gpu/acceleration_test.cpp share
// (Acceleration: a traced ray's hit): one definition for both sides,
// fixed-width fields, counts and indices as integers, each size asserted on
// both sides and trivial copyability on the host
// (core/contracts/shared_layout.h gives the rules; SF.3, I.4). Included by
// that kernel and that test alone, so a change to one probe's records touches
// no other probe (SF.11; docs/architecture/change-axes.md). The kernel binds
// its output at buffer 0 and its inputs after it in the order its comment
// lists them (tests/gpu/support/probe_runner.h).

#include "core/contracts/float3.h"
#include "core/contracts/ray_mask.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// A ray traced from `origin` along the unit `direction`, carrying `mask`
// (contract 19); `reach`, how far its shadow-ray twin looks.
struct TraceRay {
  contracts::Float3 origin;
  contracts::Float3 direction;
  contracts::RayMask mask;
  float reach;
};

static_assert(sizeof(TraceRay) == 32,
              "TraceRay must be the same 32 bytes on the host and in shaders");

// What a ray hit: its distance and its shape, or found 0; and whether a
// shadow ray along it, as far as its reach, carrying the same mask, is
// blocked (trace.metal.h, occluded()).
struct TraceHit {
  float t;
  uint32_t shape;
  uint32_t found;
  uint32_t occluded;
};

static_assert(sizeof(TraceHit) == 16,
              "TraceHit must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<TraceRay>,
              "TraceRay is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<TraceHit>,
              "TraceHit is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
