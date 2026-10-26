#pragma once

// The records mnee_probe.metal and tests/gpu/mnee_test.cpp share
// (Integrator: manifold next event estimation's solve): one definition for
// both sides, fixed-width fields, each size asserted on both sides and
// trivial copyability on the host (core/contracts/shared_layout.h gives the
// rules; SF.3, I.4). Included by that kernel and that test alone (SF.11).
// The kernel binds its output at buffer 0 and its queries at 1
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// A receiver's point, a light's point and the light's outward normal
// there, and one glass sphere between them: its center, radius and index;
// the receiver's normal on the side the receiver's point lies; whether the
// walk is taken in the MNEE pass's rounds, kept in contract 16's record
// between them (1), or whole (0).
struct MneeQuery {
  contracts::Float3 receiver;
  float ior;
  contracts::Float3 light;
  float radius;
  contracts::Float3 center;
  uint32_t in_rounds;
  contracts::Float3 light_normal;
  uint32_t padding1;
  contracts::Float3 facing;
  uint32_t padding2;
};

static_assert(sizeof(MneeQuery) == 80,
              "MneeQuery must be the same 80 bytes on the host and in shaders");

// The solved chain's two crossings, whether it was found, and its
// generalized geometry term G(x0 <-> y): solid angle at the receiver per
// area at the light, |cos at x1| / |x1 - x0|^2 |det dx1 / dy|.
struct MneeChain {
  contracts::Float3 first;
  uint32_t found;
  contracts::Float3 second;
  float geometry;
};

static_assert(sizeof(MneeChain) == 32,
              "MneeChain must be the same 32 bytes on the host and in shaders");

// A bounce for mnee::chain_after(): the chain before it, the lobe it drew,
// and the shape it drew it at (an index into the records bound beside the
// queries, or contracts::no_primitive, as a replayed shift's first surface
// has).
struct ChainQuery {
  uint32_t chain;
  uint32_t lobe;
  uint32_t primitive;
  uint32_t padding;
};

static_assert(
    sizeof(ChainQuery) == 16,
    "ChainQuery must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<ChainQuery>,
              "ChainQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<MneeQuery>,
              "MneeQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<MneeChain>,
              "MneeChain is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
