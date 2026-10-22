#pragma once

#include <cuda.h>

#include <array>
#include <cstdint>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "cuda/device/buffer.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (ReSTIR PT's arrays: contracts 5, 9, 15).
//
// The arrays ReSTIR PT's passes read and write, Metal's PathReuseImages
// (metal/frame/frame_images.h, which says what each holds and why one set
// serves every frame in flight), at the core's sizes (core/passes/
// restir_pt_arrays.h) in device memory, which only the GPU touches but the
// shifts' lists' counts, zeroed once and set back after each reuse's
// shifts on the stream (cuda/passes/restir_pt/shift_replay.h); and beside
// them CUDA's list of the pairs that reconnect, the replay list's size.
//
// Which half of a ping-pong is this frame's is its index's parity
// (passes::this_frame_copy). Which frames the arrays hold is the core's
// rule (frame::ReuseHistory), with temporal reuse only, kept by the one
// this object owns, as Accumulation keeps frame::History: prepare() asks
// it whether the frame joins, refuses by CudaError with the rule's reason
// what it refuses, and returns the frames it holds before this one. Where
// the history starts over at another size, the arrays are made anew, first
// finishing the frames in flight, which may still read the old ones;
// before the frame's submission begins. Nothing in them is cleared when
// made: a frame that starts the history over reads no last frame's half,
// and every pass writes every pixel of what it writes. The light tiles and
// the power pyramids, whose sizes depend on the scene's lights alone, are
// made once, at construction.
//
// Throws CudaError if the memory cannot be made, or the core's history
// rule refuses a frame.
class PathReuseImages {
 public:
  enum class Spatial : std::uint8_t { none, merged };
  enum class Temporal : std::uint8_t { none, merged };
  // the grid the arrays are indexed by (contract 20)
  enum class Grid : std::uint8_t { image, checkerboard };

  // `lights`: the scene's light records, which the power pyramids hold.
  PathReuseImages(Submission& submission, Spatial spatial, Temporal temporal,
                  Grid grid, std::uint32_t lights);

  PathReuseImages(const PathReuseImages&) = delete;
  PathReuseImages& operator=(const PathReuseImages&) = delete;
  PathReuseImages(PathReuseImages&&) = delete;
  PathReuseImages& operator=(PathReuseImages&&) = delete;
  ~PathReuseImages() = default;

  // Ready for frame `inputs` at `size`: the arrays made or remade, and the
  // frames the history holds before this one (always 0 without temporal
  // reuse).
  std::uint64_t prepare(const frame::FrameInputs& inputs, frame::Extent size);

  // The arrays frame `index` reads and writes: this frame's halves and
  // last frame's by its parity; 0 for what the graph does not have.
  struct Addresses {
    CUdeviceptr reservoirs = 0;
    CUdeviceptr primaries = 0;
    CUdeviceptr last_primaries = 0;
    CUdeviceptr light = 0;
    CUdeviceptr shifts = 0;
    CUdeviceptr reconnects = 0;
    CUdeviceptr reconnect_counter = 0;
    CUdeviceptr replays = 0;
    CUdeviceptr replay_counter = 0;
    CUdeviceptr light_tiles = 0;
    CUdeviceptr light_power = 0;
    CUdeviceptr last_light_power = 0;
    CUdeviceptr finals = 0;
    CUdeviceptr last_finals = 0;
    CUdeviceptr previous_positions = 0;
    CUdeviceptr previous_pixels = 0;
    CUdeviceptr duplication = 0;
  };
  Addresses addresses(std::uint64_t index) const noexcept;

 private:
  Submission& submission_;
  Spatial spatial_;
  Temporal temporal_;
  Grid grid_;
  frame::Extent size_;
  frame::ReuseHistory history_;  // asked only with temporal reuse
  DeviceBuffer reservoirs_;
  // [0] for even frames; [0] alone without temporal reuse
  std::array<DeviceBuffer, 2> primaries_;
  DeviceBuffer light_;
  DeviceBuffer shifts_;
  // The shifts' lists of pairs (cuda/passes/restir_pt/shift_replay.h), each
  // the core's replay list's size, the pairs that reconnect a CUDA list of
  // its own beside the core's arrays.
  DeviceBuffer reconnects_;
  DeviceBuffer replays_;
  // The two lists' counts, zeroed once: the reconnections' (word 0), the
  // replays' (word 1).
  DeviceBuffer counters_;
  DeviceBuffer light_tiles_;  // contracts/light_tile.h: whatever the size
  // [0] for even frames; [0] alone without temporal reuse
  std::array<DeviceBuffer, 2> light_power_;
  std::array<DeviceBuffer, 2> finals_;
  DeviceBuffer previous_positions_;
  DeviceBuffer previous_pixels_;
  DeviceBuffer duplication_;
};

}  // namespace serenity::cuda
