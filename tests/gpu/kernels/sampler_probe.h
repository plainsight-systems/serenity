#pragma once

// The records sampler_probe.metal and tests/gpu/sampler_test.cpp share
// (Sampler: a path's numbers): one definition for both sides, fixed-width
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

// A pixel and a frame whose path's numbers are drawn.
struct PathNumbersQuery {
  uint32_t x;
  uint32_t y;
  uint32_t frame;
  uint32_t padding;
};

static_assert(
    sizeof(PathNumbersQuery) == 16,
    "PathNumbersQuery must be the same 16 bytes on the host and in shaders");

// How many of a path's numbers a probe draws.
SERENITY_CONSTANT uint32_t path_numbers_drawn = 8u;

// The first path_numbers_drawn numbers of one path.
struct PathNumbersDraw {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): MSL has no std::array
  float u[path_numbers_drawn];
};

static_assert(
    sizeof(PathNumbersDraw) == 32,
    "PathNumbersDraw must be the same 32 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<PathNumbersQuery>,
              "PathNumbersQuery is copied to the GPU as bytes");
static_assert(std::is_trivially_copyable_v<PathNumbersDraw>,
              "PathNumbersDraw is copied from the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
