#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "core/passes/tone_map.h"
#include "metal/device/device.h"
#include "metal/device/resident.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (the images between passes: contract 6).
//
// The images a frame's passes hand one another within the frame, as the
// schedule says they do (core/frame/schedule.h):
//
//   - the radiance image: the frame's linear radiance, RGBA32Float at the
//     frame's size, written by the denoise pass at the output's size and
//     read by the pass that presents it (tone_map). 32-bit, not 16: a
//     firefly's radiance of hundreds keeps its digits. A light pass that
//     accumulates (path, restir_pt_shade) in a graph without the denoise
//     pass has none here: its accumulated image, the same format at the
//     same size, is the frame's radiance (accumulation.h, passes/path/
//     path.h), so the mean is written once (GDSA.6);
//   - the bloom pyramid, for the tone-map pass alone: passes::bloom_levels
//     levels (core/passes/tone_map.h), each max(1, ceil(previous / 2)) on
//     each axis from the frame's size, so any frame has every level;
//     RGBA16Float, which a blur of light needs no more than (half's 11 bits
//     are 3 decimal digits), the light kept within its range
//     (core/passes/tone_map.h, steps 1 to 3).
//
// Made when the frame's size is first known and remade when it changes,
// before the frame's submission begins (Renderer::prepare), draining the
// submission first, as the accumulated image is (accumulation.h): frames in
// flight may still read the old images. Private storage, the GPU's alone,
// made resident through the submission. Only what the schedule uses is
// made: a schedule with no light pass makes none; one without tone_map, no
// pyramid.
//
// One set, not one per frame in flight: the frames in flight share it, and
// what keeps one frame's passes from writing it while the frame before still
// reads it is a barrier the renderer records before the first pass of a
// frame that writes it, waiting for the queue's earlier dispatches
// (metal/frame/renderer.h). So the CPU still records a frame while the GPU
// runs the one before; what the barrier gives up is only the GPU starting a
// frame's first dispatch before the last frame's final one ends. A copy per
// frame in flight would keep that sliver of overlap at twice the memory,
// 290 MB at the display's size, and the overlap is no gain: two frames'
// passes running at once take longer than one after the other (measured,
// docs/research/2026-10-09-pass-costs.md).
//
// Made outside the frame loop, at a size, and kept until the size changes
// (GPU.9); nothing else lives only within a frame for them to share memory
// with yet. The radiance image is an intermediate the frame writes and reads
// back, which GDSA.16 says to stream through on-chip memory instead: the
// tone map cannot, since its bloom reads every pixel's neighbourhood as far
// as B_5's 64 pixels.
//
// Cost: at 3456 x 2234 (P = 7.7 M pixels), the radiance image is 123 MB, in
// a denoised graph, and the pyramid 21 MB. Per frame, the denoise pass
// writes the radiance image once, P texels. The tone-map pass reads it twice,
// in two dispatches ten apart, too far for one read to leave it cached for the
// other: 13 filtered samples for each of B_0's P / 4 texels, then P exact
// reads (passes/tone_map/tone_map.h); what that costs in memory traffic is
// the pass's measured time, not a count here.
//
// Throws MetalError if the frame's size is not one Metal makes an image of
// (metal/device/device.h, max_texture_side), or the device cannot make one.
class FrameImages {
 public:
  // Whether the frame's radiance is an image of its own, or the light
  // pass's accumulated image (accumulation.h), which this then does not
  // make. Types, not bools, so the call says which (I.4).
  enum class Radiance : std::uint8_t { image, accumulated };

  // Whether the schedule has the tone-map pass, which needs the pyramid.
  enum class Bloom : std::uint8_t { none, pyramid };

  // The radiance image as `radiance` says, and the pyramid as `bloom`
  // says (a schedule with no light pass makes no FrameImages).
  FrameImages(const Device& device, Submission& submission, Radiance radiance,
              Bloom bloom);

  FrameImages(const FrameImages&) = delete;
  FrameImages& operator=(const FrameImages&) = delete;
  FrameImages(FrameImages&&) = delete;
  FrameImages& operator=(FrameImages&&) = delete;
  ~FrameImages() = default;

  // Ready for a frame of `size`: made, or remade, if not made at it.
  void prepare(frame::Extent size);

  // Null when the schedule does not use it, or before the first prepare().
  // bloom() throws MetalError for a level past passes::bloom_levels.
  MTL::Texture* radiance() const noexcept { return radiance_.texture.get(); }
  MTL::Texture* bloom(std::uint32_t level) const;

 private:
  NS::SharedPtr<MTL::Device> device_;
  Submission& submission_;
  Radiance radiance_kind_;
  Bloom bloom_;
  frame::Extent size_;
  // Each image with its residency, released after the GPU is done with it
  // (submission.h, Lifetime).
  ResidentTexture radiance_;
  std::array<ResidentTexture, passes::bloom_levels> pyramid_;
};

// ReSTIR PT's arrays between its passes and across frames (metal/passes/
// restir_pt/restir_pt.metal.h states what each pass reads and writes of
// them), a graph with its passes only, private to the GPU:
//
//   reservoirs          a path reservoir a pixel (contract 15), 64 bytes:
//                       the initial pass's, merged in place by the temporal
//                       reuse where a spatial reuse follows;
//   primaries           each pixel's primary hit (contract 5's PrimaryHit),
//                       48 bytes; two, ping-ponged, with temporal reuse:
//                       this frame's and last frame's (MEM.4);
//   light               the frame's light buffer, a float4 a pixel, 16 bytes;
//   light tiles         the frame's (contracts/light_tile.h), 4 MB whatever
//                       the size;
//   power pyramids      each frame's lights' powers and their sums, which
//                       the tiles are drawn from (contracts/light_tile.h),
//                       light_power_floats() floats for the scene's lights,
//                       5.3 KB for 616; two, ping-ponged by the frame's
//                       parity, with temporal reuse, whose shifts to last
//                       frame's pixel weigh by last frame's;
//
// with restir_pt_spatial or restir_pt_temporal:
//
//   shifts              each pixel's path shifted to each of its 3 partners,
//                       or the temporal reuse's two shifts, F and the
//                       Jacobian, 48 bytes;
//   replays             the shifts that replay, at most 3 a pixel, 12 bytes;
//   counter             their count and the replay's indirect grid, in one
//                       small buffer the CPU zeroes once (shared storage, 128
//                       bytes): the grid pass sets the count back to 0;
//
// and with restir_pt_temporal:
//
//   finals              the frame's final reservoirs, 64 bytes, two,
//                       ping-ponged: what the last merge writes and the next
//                       frame's temporal reuse reads as last frame's;
//   previous positions  where each pixel's primary hit was last frame, a
//                       Float3, 12 bytes;
//   previous pixels     the temporal input's pixel, 4 bytes;
//   duplication         the duplication map, a float a pixel: written after
//                       a frame, read by the next frame's temporal reuse
//                       before this frame's map is written over it, so one.
//
// Which half of a ping-pong is this frame's is the frame's index's parity:
// the reuse history holds only consecutive frames (core/frame/history.h),
// so last frame's half is always the other.
//
// The history. Which frames the arrays hold is the core's rule
// (frame::ReuseHistory), with temporal reuse only, kept by the one this
// object owns, as Accumulation keeps frame::History (accumulation.h):
// prepare() asks it whether the frame joins, refuses by MetalError with the
// rule's reason what it refuses, and returns the frames it holds before
// this one, 0 where it starts over (the temporal reuse then merges
// nothing). Where the rule says the history starts over at another size,
// the arrays are made anew: first draining the submission, as the
// accumulated image is remade, since frames in flight may still read the
// old ones; before the frame's submission begins (Renderer::prepare).
// Private storage, the GPU's alone, made resident through the submission,
// made outside the frame loop and kept until the size changes (GPU.9,
// MEM.9). Nothing in them is cleared when made: a frame that starts the
// history over reads no last frame's half, and every pass writes every
// pixel of what it writes.
//
// One set, not one per frame in flight, as FrameImages: the renderer's
// queue barrier before the initial pass keeps a frame from writing what
// the frame before still reads (metal/frame/renderer.h).
//
// Bytes a pixel: 128 for the initial pass alone; 188 with spatial reuse;
// with both reuses, 64 + 2 x 48 + 16 + 48 + 12 + 2 x 64 + 12 + 4 + 4 = 384,
// 2.96 GB at 3456 x 2234 and 741 MB at 1728 x 1117.
//
// Throws MetalError if the frame's size is not one Metal makes an image of,
// the device cannot make a buffer, or the core's history rule refuses a
// frame.
class PathReuseImages {
 public:
  enum class Spatial : std::uint8_t { none, merged };
  enum class Temporal : std::uint8_t { none, merged };

  // `lights`: the scene's light records, which the power pyramids hold.
  PathReuseImages(const Device& device, Submission& submission, Spatial spatial,
                  Temporal temporal, std::uint32_t lights);

  PathReuseImages(const PathReuseImages&) = delete;
  PathReuseImages& operator=(const PathReuseImages&) = delete;
  PathReuseImages(PathReuseImages&&) = delete;
  PathReuseImages& operator=(PathReuseImages&&) = delete;
  ~PathReuseImages() = default;

  // Ready for frame `inputs` at `size`: the arrays made or remade, and
  // the frames the history holds before this one (always 0 without
  // temporal reuse).
  std::uint64_t prepare(const frame::FrameInputs& inputs, frame::Extent size);

  // The arrays frame `index` reads and writes: this frame's halves and
  // last frame's by its parity; 0 for what the graph does not have.
  struct Addresses {
    MTL::GPUAddress reservoirs = 0;
    MTL::GPUAddress primaries = 0;
    MTL::GPUAddress last_primaries = 0;
    MTL::GPUAddress light = 0;
    MTL::GPUAddress shifts = 0;
    MTL::GPUAddress replays = 0;
    MTL::GPUAddress counter = 0;
    MTL::GPUAddress grid = 0;
    MTL::GPUAddress light_tiles = 0;
    MTL::GPUAddress light_power = 0;
    MTL::GPUAddress last_light_power = 0;
    MTL::GPUAddress finals = 0;
    MTL::GPUAddress last_finals = 0;
    MTL::GPUAddress previous_positions = 0;
    MTL::GPUAddress previous_pixels = 0;
    MTL::GPUAddress duplication = 0;
  };
  Addresses addresses(std::uint64_t index) const noexcept;

  // Where the indirect grid is in the counter's buffer: past the count,
  // at its own alignment (metal/device/indirect_grid.h) and the constant
  // offset alignment (support.h), which every binding meets; the buffer's
  // bytes, the grid within them.
  static constexpr std::size_t grid_offset = 64;
  static constexpr std::size_t counter_bytes = 128;

 private:
  using Array = ResidentBuffer;  // metal/device/resident.h

  void make(Array& array, std::size_t bytes, std::string_view what);

  NS::SharedPtr<MTL::Device> device_;
  Submission& submission_;
  Spatial spatial_;
  Temporal temporal_;
  frame::Extent size_;
  frame::ReuseHistory history_;  // asked only with temporal reuse
  Array reservoirs_;
  std::array<Array, 2>
      primaries_;  // [0] for even frames; [0] alone without temporal reuse
  Array light_;
  Array shifts_;
  Array replays_;
  Array counter_;      // shared storage, zeroed once
  Array light_tiles_;  // contracts/light_tile.h: a frame's, whatever its size
  std::array<Array, 2>
      light_power_;  // [0] for even frames; [0] alone without temporal reuse
  std::array<Array, 2> finals_;
  Array previous_positions_;
  Array previous_pixels_;
  Array duplication_;
};

}  // namespace serenity::metal
