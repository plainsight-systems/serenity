#pragma once

// Contract 20: checkerboard rendering of the light. Owned by the Frame
// graph; read by Sample reuse and Pass: ReSTIR PT's kernels, the fill
// (metal/passes/checkerboard_fill/checkerboard_fill.h) and the host that
// dispatches them (architecture/logical-overview.md, "The frame";
// Milestone 4's Step 4, docs/research/2026-10-17-milestone-4-plan.md).
//
// In a graph whose light is the network's input (core/frame/schedule.h,
// feeds_network: the denoised graph and the data set's input graph), frame
// n traces
// the light at the pixels (x, y) of the light size whose x + y + n
// is even, half of them, a checkerboard whose two fields alternate frame by
// frame: so every pixel is traced in every other frame, and in one of any
// two frames in a row. This is RTXDI's checkerboard mode, its "black"
// field order (RTXDI-Library, Rtxdi/Utils/Checkerboard.hlsli and
// Source/ReSTIRDI.cpp: field 1 on odd frames, the pixels whose x + y is
// odd; field 2 on even frames, even x + y), stated as one parity.
//
// Where a traced pixel's arrays keep it: ReSTIR PT's per-pixel arrays (its
// path reservoirs, primary hits, light buffer, previous positions and
// pixels, shifted paths and duplication map) are packed, as RTXDI packs its
// reservoirs (Rtxdi/Utils/ReservoirAddressing.hlsli,
// RTXDI_PixelPosToReservoirPos): the pixels x = 2k and 2k + 1 of a row
// share slot k of that row, a row checkerboard_width() slots long. In each
// frame exactly one of the two is traced, so a slot holds the pixel its
// frame traced, and the frame after reads it as that pixel's (the
// temporal input, below). The traced half then lies side by side, so a
// SIMD group's threads read and write adjacent slots (GPU.2: coalesced
// lane access), where a full-size array indexed by pixel would leave every
// other element between them unread.
//
// How the arrays store the grid they are indexed by (the packed one, or
// the image without the checkerboard): RTXDI's second step
// (RTXDI_ReservoirPositionToPointer) stores it in blocks of
// reservoir_block x reservoir_block slots, each block's 256 slots
// contiguous, row by row within it, and the blocks row by row
// (reservoir_slot), so a 16 x 16 group of threads writes one contiguous
// span and a neighborhood lies in a few blocks, not across as many of the
// grid's rows (GPU.2). CUDA's kernels address the arrays so (cuda/passes/
// restir_pt/restir_pt.cuh, Traced; docs/research/2026-10-20-nvidia-
// hardware-design.md, section 8, change 9); Metal's by the grid's rows
// (checkerboard_slot), its own GPU's layout. Either fits the arrays' size,
// reservoir_slots(), the grid in whole blocks (core/passes/
// restir_pt_arrays.h).
//
// The grid the light's kernels dispatch over is the packed one,
// checkerboard_width() x height: thread (k, y) is the pixel
// (checkerboard_x(k, y, n), y), always one frame n traces, so no lane of a
// SIMD group idles on a pixel it skips (GPU.4: coherent control flow), as
// a full grid with half its threads returning would. A thread whose pixel
// falls past an odd width's last column holds none, as a thread past the
// image's edge always has.
//
// The frame before's traced pixel beside a pixel (checkerboard_previous_x):
// itself if frame n - 1 traced it, else its neighbor in the row, the one to
// its left in an odd frame and to its right in an even one, as RTXDI's
// RTXDI_ActivateCheckerboardPixel moves a temporal candidate into the
// previous field. Where that neighbor is past the image's edge the caller
// finds no candidate there, as for any try outside the image.
//
// The traced field is a lattice: with phi = n mod 2, its pixels are
// (u + v + phi, u - v) over every integer u and v, one step of it
// checkerboard_lattice_step pixels long. ReSTIR PT's paired spatial reuse
// pairs within it (metal/reuse/pairing.metal.h), its reuse textures built
// at sigma over that step so a partner lies sigma pixels away along each
// axis, as without it.
//
// Every function here is pure, a function of its arguments alone (F.8): the
// field is the frame's index, which every pass reads from the frame
// constants (contract 7), so the host and every kernel agree about which
// pixel a thread or a slot is by construction (ES.3: one definition), and
// any frame renders again alike (logical-overview.md, principle 2). The
// field's arithmetic wraps with the index at 2^32 frames, an even count,
// so the parity holds across the wrap.
//
// The fill's constants (CheckerboardConstants): whether the fill's history
// (metal/frame/checkerboard_images.h) holds this history's frame before,
// at this size, which the fill reads; the renderer writes them each frame
// (metal/frame/renderer.h).
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace contracts {

// The length of one step of the traced field's lattice, in pixels: the
// diagonal, sqrt(2) (above).
SERENITY_CONSTANT float checkerboard_lattice_step = 1.41421356f;

// Whether frame `frame` traces the pixel (x, y).
SERENITY_SHARED_FUNCTION constexpr bool checkerboard_traced(uint32_t x,
                                                            uint32_t y,
                                                            uint32_t frame) {
  return ((x + y + frame) & 1u) == 0u;
}

// The packed grid's width, in slots, for an image `width` pixels wide.
SERENITY_SHARED_FUNCTION constexpr uint32_t checkerboard_width(uint32_t width) {
  return width / 2u + (width & 1u);
}

// The pixel's x that column `column` of row `y` of the packed grid holds in
// frame `frame`: the one of 2 column and 2 column + 1 that the frame
// traces. Possibly the width itself, past an odd width's last column.
SERENITY_SHARED_FUNCTION constexpr uint32_t checkerboard_x(uint32_t column,
                                                           uint32_t y,
                                                           uint32_t frame) {
  return 2u * column + ((y + frame) & 1u);
}

// The packed slot of the pixel (x, y) of an image `width` pixels wide, in
// either field.
SERENITY_SHARED_FUNCTION constexpr uint32_t checkerboard_slot(uint32_t x,
                                                              uint32_t y,
                                                              uint32_t width) {
  return y * checkerboard_width(width) + x / 2u;
}

// The side of a block of the arrays' slots (above).
SERENITY_CONSTANT uint32_t reservoir_block = 16u;

// The blocks along a side of the arrays' grid `extent` slots long.
SERENITY_SHARED_FUNCTION constexpr uint32_t reservoir_blocks(uint32_t extent) {
  return (extent + reservoir_block - 1u) / reservoir_block;
}

// The slots the arrays hold for a grid of `grid_width` x `height`: the
// grid in whole blocks.
SERENITY_SHARED_FUNCTION constexpr uint32_t reservoir_slots(uint32_t grid_width,
                                                            uint32_t height) {
  return reservoir_blocks(grid_width) * reservoir_blocks(height) *
         reservoir_block * reservoir_block;
}

// The block-linear slot of the place (x, y) of a grid `grid_width` slots
// wide (above).
SERENITY_SHARED_FUNCTION constexpr uint32_t reservoir_slot(
    uint32_t x, uint32_t y, uint32_t grid_width) {
  return ((y / reservoir_block) * reservoir_blocks(grid_width) +
          x / reservoir_block) *
             (reservoir_block * reservoir_block) +
         (y % reservoir_block) * reservoir_block + x % reservoir_block;
}

// The x of the pixel frame `frame - 1` traced beside the pixel (x, y)
// (above): x itself, or x - 1 in an odd frame and x + 1 in an even one.
// Signed: x - 1 may be -1, outside the image, which the caller tests.
SERENITY_SHARED_FUNCTION constexpr int32_t checkerboard_previous_x(
    int32_t x, int32_t y, uint32_t frame) {
  if (checkerboard_traced(static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                          frame - 1u)) {
    return x;
  }
  return (frame & 1u) != 0u ? x - 1 : x + 1;
}

// What the fill reads beyond the frame's constants (above).
struct CheckerboardConstants {
  // 1 when the fill's history holds this history's frame before
  uint32_t history;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): MSL has no std::array
  uint32_t padding[3];
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): Metal's layout
static_assert(sizeof(CheckerboardConstants) == 16,
              "CheckerboardConstants must be the same 16 bytes on the host and "
              "in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<CheckerboardConstants>,
              "CheckerboardConstants is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
