#pragma once

#include <cstddef>
#include <cstdint>

#include "core/frame/extent.h"

namespace serenity::passes {

// Axis: Sample reuse (ReSTIR PT's arrays and their sizes, for every
// backend).
//
// Which arrays ReSTIR PT's passes read and write beyond the frame's images,
// and how many bytes each holds, for each kind of reuse a graph has
// (contracts 5, 9, 15, 16; their uses: metal/frame/frame_images.h,
// PathReuseImages). A backend makes them at these sizes, in its own
// memory, and gives each pass their addresses; which it makes and how big
// they are is the same on every backend, so decided here (A.1), each a
// pure function (F.8).
//
// A slot each of the arrays' grid, the frame's pixels or, with the
// checkerboard, the packed grid of the half each frame traces (contract
// 20, core/contracts/checkerboard.h), in whole blocks of its slots
// (reservoir_slots(), RTXDI's): the path reservoirs (contract 15), the
// primary hits (contract 5), the light buffer (a float4); with spatial or
// temporal reuse, the shifted paths (contracts::max_path_neighbors float4s)
// and the replaying shifts (as many uints); with temporal reuse, the final
// reservoirs, the previous positions (a Float3), the previous pixels (a
// uint) and the duplication map (a float). The primary hits, the final
// reservoirs and the power pyramid are ping-ponged with temporal reuse, two
// copies, this frame's by its index's parity (this_frame_copy); one copy
// without it. An array a graph does not have is 0 bytes.
struct PathReuseArrays {
  std::size_t reservoirs = 0;
  std::size_t primaries = 0;  // each copy
  std::size_t light = 0;
  std::size_t shifts = 0;
  std::size_t replays = 0;
  std::size_t finals = 0;  // each copy
  std::size_t previous_positions = 0;
  std::size_t previous_pixels = 0;
  std::size_t duplication = 0;
  std::size_t copies = 1;  // of the ping-ponged arrays
};

PathReuseArrays path_reuse_arrays(frame::Extent size, bool spatial,
                                  bool temporal, bool checkerboard);

// The light tiles (contract 9), whatever the frame's size.
std::size_t light_tiles_bytes();

// One copy of the power pyramid over `lights` light records (contract 9,
// light_power_levels).
std::size_t light_power_bytes(std::uint32_t lights);

// Which copy of a ping-ponged array is frame `index`'s: its parity with
// temporal reuse, the one copy without; last frame's is the other, since
// the reuse history holds only consecutive frames (core/frame/history.h).
constexpr std::size_t this_frame_copy(std::uint64_t index,
                                      bool temporal) noexcept {
  return temporal ? index % 2 : 0;
}

}  // namespace serenity::passes
