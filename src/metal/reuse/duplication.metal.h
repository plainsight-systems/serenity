#pragma once

// Axis: Sample reuse (ReSTIR PT's duplication map).
//
// ReSTIR PT Enhanced's decorrelation (Lin, Kettunen and Wyman 2026, §5):
// after a frame, each pixel counts how many final reservoirs in its 17 x 17
// neighborhood hold copies of its own path, D = count / 288 in [0, 1]; the
// next frame's temporal merge lowers the cap on the temporal input it reads
// at a pixel by that pixel's D (metal/reuse/path_merge.metal.h,
// duplication_cap). A copy is a path shifted from the same initial
// candidate: the same numbers' stream, the reservoir's (seed pixel, seed
// frame) (contract 15), as Enhanced detects it ("comparing the random seeds
// already stored for random replay"). Copies are what spatial reuse makes
// of a bright path, and what temporal reuse then keeps: where many pixels
// hold one, it is the blotch a viewer sees, and its cap falls toward
// cap_floor. An empty reservoir (no path, or a weight of 0) holds no path
// and is no one's copy: its D is 0.
//
// One threadgroup of 24 x 24 pixels (contracts::duplication_group) loads
// the keys of its pixels and the 8 around them, 40 x 40, into threadgroup
// memory (12.5 KB, under the 32 KB Metal allows a group with its shader
// validation's shadow copy, which a 32 x 32 group's 18 KB was not), each
// thread two or three; then each pixel compares its key with the 289 of its
// neighborhood, itself included and taken back out (GDSA.16: a gather over
// a neighborhood streamed through on-chip memory, the reservoirs read 2.8
// times a pixel rather than 289). Exact: the whole 64-bit stream is
// compared, so two paths are counted alike only when they are copies.
//
// Cost, per pixel: 2.8 reads of 12 bytes of a reservoir (its weight, seed
// and flags, in its first 32-byte half), 289 comparisons of 8 bytes in
// threadgroup memory, a 4-byte write. Measured in docs/research/
// 2026-10-11-restir-pt-phase2.md.

#include <metal_stdlib>

#include "core/contracts/checkerboard.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"

namespace serenity {
namespace shaders {
namespace duplication {

constant constexpr uint radius = 8u;
constant constexpr uint window = 2u * radius + 1u;  // 17 x 17
// pixels on a side of a threadgroup
constant constexpr uint group = serenity::contracts::duplication_group;
constant constexpr uint span = group + 2u * radius;
// the window but its pixel: 288
constant constexpr float others = float(window * window - 1u);
// no pixel key is: x + 2^16 y with x, y under 16384
constant constexpr uint2 no_key = uint2(0xffffffffu);

// The key a reservoir is counted by (above): its stream, or no_key for an
// empty one.
inline uint2 key(serenity::contracts::PathReservoir reservoir) {
  const bool holds = serenity::contracts::path_end(reservoir.flags) !=
                         serenity::contracts::PathEnd::none &&
                     reservoir.weight > 0.0f;
  return holds ? uint2(reservoir.seed_pixel, reservoir.seed_frame) : no_key;
}

// The group's keys into `keys`, span x span, the group's first pixel
// `origin`, by its thread `thread_index` of group x group; no_key outside
// the grid of `size`, whose place (x, y) is at the slot
// contracts::reservoir_slot(x, y, size.x) (contract 20).
inline void load_keys(
    threadgroup uint2* keys,
    const device serenity::contracts::PathReservoir* reservoirs, uint2 size,
    uint2 origin, uint thread_index) {
  for (uint n = thread_index; n < span * span; n += group * group) {
    const int2 at = int2(origin) - int2(radius) + int2(n % span, n / span);
    const bool inside =
        metal::all(at >= int2(0)) && metal::all(at < int2(size));
    keys[n] = inside ? key(reservoirs[serenity::contracts::reservoir_slot(
                           uint(at.x), uint(at.y), size.x)])
                     : no_key;
  }
}

// D at the pixel `local` within the group (above), from the loaded keys.
inline float score(threadgroup const uint2* keys, uint2 local) {
  const uint2 own = keys[(local.y + radius) * span + local.x + radius];
  if (metal::all(own == no_key)) {
    return 0.0f;
  }
  uint count = 0u;
  for (uint dy = 0u; dy <= 2u * radius; ++dy) {
    for (uint dx = 0u; dx <= 2u * radius; ++dx) {
      count += metal::all(keys[(local.y + dy) * span + local.x + dx] == own)
                   ? 1u
                   : 0u;
    }
  }
  return float(count - 1u) / others;
}

}  // namespace duplication
}  // namespace shaders
}  // namespace serenity
