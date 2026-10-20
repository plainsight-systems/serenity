#pragma once

// Axis: Pass (path tracing: what its launch is given).
//
// The path pass's launch parameters (path.h): one struct in OptiX's
// launch-parameter variable, which the programs read from constant memory,
// the space Metal's `constant` buffers are in (docs/research/2026-10-19-
// cuda-backend.md, "Phase 3's design"). Written by the host each frame into
// the slot's staging copy and uploaded on the frame's stream
// (cuda/device/frame_upload.h); read by path.cu as `launch`. The frame's
// constants and the camera by value, as Metal binds them; the structures'
// handles; and the addresses of what the pass reads and writes. Addresses
// on the host, pointers in the programs (gpu::Array, core/scene/
// scene_block.h, for what is read; Writes, below, for what is written).
// Compiled by both the host compiler and nvcc: one type.

#include <cstdint>

#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
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

struct PathLaunch {
  contracts::FrameConstants frame;
  contracts::CameraData camera;
  StructureBlock structures;
  gpu::Array<gpu::SceneBlock> scene;
  gpu::Array<contracts::Transform> transforms;  // as this frame places them
  gpu::Array<float> glows;                      // as this frame lights them
  Writes<std::uint32_t> non_finite;             // the slot's counter
  // the accumulated image's row 0, four floats a pixel, rows `pitch` bytes
  // apart (cuda/device/images.h, PitchedImage)
  Writes<std::uint8_t> accumulated;
  std::uint64_t pitch;
};

}  // namespace serenity::cuda
