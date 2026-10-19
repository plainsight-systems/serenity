#pragma once

// The records scene_block_probe.metal and tests/gpu/scene_block_test.cpp
// share (the scene block's arrays): one definition for both sides,
// fixed-width fields, counts and indices as integers, each size asserted on
// both sides and trivial copyability on the host
// (core/contracts/shared_layout.h gives the rules; SF.3, I.4). Included by
// that kernel and that test alone, so a change to one probe's records touches
// no other probe (SF.11; docs/architecture/change-axes.md). The kernel binds
// its output at buffer 0 and its inputs after it in the order its comment
// lists them (tests/gpu/support/probe_runner.h).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// A 32-bit word of an element of one of the scene block's arrays: field i
// of gpu::SceneBlock, in the order it declares them.
struct BlockQuery {
  uint32_t field;
  uint32_t element;
  uint32_t word;
};

static_assert(
    sizeof(BlockQuery) == 12,
    "BlockQuery must be the same 12 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<BlockQuery>,
              "BlockQuery is copied to the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
