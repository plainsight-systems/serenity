#pragma once

// Axis: Frame graph (what every pass of a frame is given).
//
// The frame's launch parameters: Metal's frame resources (metal/frame/
// frame_resources.h) as one struct the passes read on the GPU
// (docs/research/2026-10-19-cuda-backend.md, "Phase 4's design"). The
// renderer writes it once a frame into the frame's slot and uploads it once
// on the frame's stream (cuda/device/frame_upload.h; GPU.6: one copy, not
// one a pass); every OptiX pass launches with it, as the variable `launch`
// its programs declare, which OptiX reads from constant memory, the space
// Metal's `constant` buffers are in; every CUDA kernel takes its address.
// Each pass reads what it needs and nothing else; a field a graph does not
// have is 0, and no pass of that graph reads it (the renderer refuses a
// pass whose arrays the frame lacks before it records the frame).
//
// The frame's constants, camera and the contracts' small per-frame and
// per-graph blocks by value, as Metal binds them in its constant space; the
// arrays by address: on the host a device address, in device code a
// pointer (gpu::Array for what the passes read, core/scene/scene_block.h;
// Writes, below, for what one writes). Compiled by both compilers: CUDA's
// vector types from its own header, which the host compiler reads too.

#include <vector_types.h>

#include <cstdint>

#include "core/contracts/camera.h"
#include "core/contracts/checkerboard.h"
#include "core/contracts/features.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "cuda/acceleration/structure_block.h"

namespace serenity::cuda {

#if defined(__CUDACC__)
template <typename T>
using Writes = T*;
#else
template <typename T>
using Writes = std::uint64_t;  // the memory's device address
#endif

// An image of four floats a pixel in pitched memory (cuda/device/images.h,
// PitchedImage): row 0's address and the bytes from a row to the next.
struct PitchedRows {
  Writes<std::uint8_t> rows;
  std::uint64_t pitch;
};

struct FrameLaunch {
  // The frame's constants and camera, framed for its size; the camera's
  // fields 0 when it has none.
  contracts::FrameConstants frame;
  contracts::CameraData camera;

  // The scene (cuda/scene/), all 0 without one: the structures by ray mask,
  // the block, the shapes' transforms as this frame places them and the
  // sphere lights' glows as it lights them.
  StructureBlock structures;
  gpu::Array<gpu::SceneBlock> scene;
  gpu::Array<contracts::Transform> transforms;
  gpu::Array<float> glows;

  // A converging pass's (cuda/frame/accumulation.h; cuda/film/
  // failure_count.h): the accumulated image and the slot's counter of
  // samples not finite.
  PitchedRows accumulated;
  Writes<std::uint32_t> non_finite;

  // ReSTIR PT's (cuda/frame/path_reuse_images.h): its settings, the frame's
  // grid of traced pixels (every pixel, or the checkerboard's field packed:
  // contract 20), the reuse textures and its arrays, this frame's halves and
  // last frame's by the frame's parity; with temporal reuse its constants
  // and last frame's transforms and glows. The shifts' two lists of pairs
  // (cuda/passes/restir_pt/shift_replay.h), the pairs that reconnect and
  // the pairs that replay, and their counts, where the pairing appends and
  // the shifts read them.
  contracts::PathReuseSettings path_settings;
  contracts::PathReuseConstants path_constants;
  uint2 traced;
  gpu::Array<std::uint16_t> pairing;
  Writes<contracts::PathReservoir> path_reservoirs;
  Writes<contracts::PrimaryHit> primaries;
  Writes<float4> light;
  Writes<float4> shifts;
  Writes<std::uint32_t> reconnects;
  Writes<std::uint32_t> reconnect_counter;
  Writes<std::uint32_t> replays;
  Writes<std::uint32_t> replay_counter;
  Writes<contracts::LightTileEntry> light_tiles;
  Writes<float> light_power;
  gpu::Array<float> last_light_power;
  gpu::Array<contracts::PrimaryHit> last_primaries;
  Writes<contracts::PathReservoir> path_finals;
  gpu::Array<contracts::PathReservoir> last_path_finals;
  Writes<contracts::Float3> previous_positions;
  Writes<std::uint32_t> previous_pixels;
  Writes<float> duplication;
  gpu::Array<contracts::Transform> last_transforms;
  gpu::Array<float> last_glows;

  // MNEE's (cuda/frame/mnee_list.h): its work list and its header, the
  // count and the capacity (contract 16), and the slot's counter of
  // requests the list had no room for.
  Writes<contracts::MneeRequest> mnee_requests;
  Writes<std::uint32_t> mnee_header;
  Writes<std::uint32_t> mnee_lost;
  // The solve's walks between its rounds (contract 16,
  // core/contracts/mnee_request.h): the rounds' two lists, the converged
  // walks', their counts.
  Writes<contracts::MneeWalk> mnee_walks_a;
  Writes<contracts::MneeWalk> mnee_walks_b;
  Writes<contracts::MneeWalk> mnee_converged;
  Writes<std::uint32_t> mnee_walk_header;

  // The features pass's (cuda/frame/feature_images.h): its constants, the
  // shapes as the frame before placed them, and the four feature images,
  // contract 17's formats: albedo, normal and motion four half floats a
  // pixel, depth one float, rows tightly packed.
  contracts::FeatureConstants feature_constants;
  gpu::Array<contracts::Transform> feature_last_transforms;
  Writes<ushort4> albedo;
  Writes<ushort4> normal;
  Writes<float> depth;
  Writes<ushort4> motion;

  // The checkerboard's fill's (cuda/frame/checkerboard_images.h): its
  // constants, the frame before's history it reads and this frame's it
  // writes, and the light it hands on, which the presenting pass shows.
  contracts::CheckerboardConstants checkerboard_constants;
  PitchedRows previous_light;
  PitchedRows history_light;
  PitchedRows filled;
};

// The same bytes from both compilers: every field 8-byte aligned, no
// padding but the contracts' own.
static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): the layout
    sizeof(FrameLaunch) == 696,
    "FrameLaunch must be the same 696 bytes on the host and in programs");

}  // namespace serenity::cuda
