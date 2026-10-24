#pragma once

// The records seed_probe.metal (and the CUDA backend's seed_probe program,
// tests/cuda/kernels/traced_probes.cu) and tests/gpu/mnee_test.cpp share
// (Integrator: manifold next event estimation's seed): one definition for
// both sides, fixed-width fields, each size asserted on both sides and
// trivial copyability on the host (core/contracts/shared_layout.h gives the
// rules; SF.3, I.4). Included by those kernels and that test alone (SF.11).
// The kernel binds its output at buffer 0 and its inputs after it in the
// order its comment lists them (tests/gpu/support/traced_probe.h).

#include "core/contracts/float3.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// A segment seeded: from `origin` along the unit `direction`, as far as
// `distance`.
struct SeedSegment {
  contracts::Float3 origin;
  float distance;
  contracts::Float3 direction;
  uint32_t padding;
};

static_assert(sizeof(SeedSegment) == 32,
              "SeedSegment must be the same 32 bytes on the host and in "
              "shaders");

// The seed found (mnee::Seed): its crossings of caster spheres, sorted,
// each's t and shape; whether another shape lies on the segment; whether
// the solve may take it. As many crossings as contract 16 takes
// (core/contracts/mnee_request.h).
struct SeedFound {
  uint32_t count;
  uint32_t blocked;
  uint32_t usable;
  uint32_t padding;
  float t[contracts::mnee_most_crossings];
  uint32_t shape[contracts::mnee_most_crossings];
};

static_assert(sizeof(SeedFound) == 48,
              "SeedFound must be the same 48 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<SeedSegment>,
              "SeedSegment is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<SeedFound>,
              "SeedFound is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
