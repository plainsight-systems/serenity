#pragma once

// The records texture_probe.metal and tests/gpu/texture_test.cpp share
// (Texture: a point's value): one definition for both sides, fixed-width
// fields, counts and indices as integers, each size asserted on both sides
// and trivial copyability on the host (core/contracts/shared_layout.h gives
// the rules; SF.3, I.4). Included by that kernel and that test alone, so a
// change to one probe's records touches no other probe (SF.11;
// docs/architecture/change-axes.md). The kernel binds its output at buffer 0
// and its inputs after it in the order its comment lists them
// (tests/gpu/support/probe_runner.h).

#include "core/contracts/float3.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace tests {

// A point a texture is evaluated at, and the noise's seed (noise_probe only).
struct TexturePoint {
  contracts::Float3 point;
  uint32_t seed;
};

static_assert(
    sizeof(TexturePoint) == 16,
    "TexturePoint must be the same 16 bytes on the host and in shaders");

#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<TexturePoint>,
              "TexturePoint is copied to the GPU as bytes");
#endif

}  // namespace tests
}  // namespace serenity
