#pragma once

#include <cstdint>
#include <optional>

#include "core/contracts/camera.h"
#include "core/contracts/checkerboard.h"
#include "core/contracts/denoise.h"
#include "core/contracts/features.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/sampling/jitter.h"

namespace serenity::frame {

// Axis: Frame graph (what each frame's constants hold, for every backend).
//
// The constants blocks a frame's passes read (contracts 7, 15, 17, 18 and
// 20), each filled from the frame's inputs and what the backend prepared
// for it, so every backend's shaders read the same numbers for the same
// frame (logical-overview.md, principle 10). Where a backend keeps them,
// a ring slot, a buffer, a launch's parameters, is its own, one copy per
// frame in flight so the CPU never writes what the GPU still reads (GPU.7;
// Metal's: metal/frame/renderer.h, "Frame constants").
//
// The core's (A.1: stable code apart from what changes with the backend),
// each a pure function of its arguments (F.8), returning the block by value
// (F.20); every field not named is 0, padding included, so a block's bytes
// are a function of its arguments too.

// Contract 7's, for frame `inputs` writing an image of `at`, joining an
// accumulated image of `accumulated_frames` (0 when it starts over or the
// graph has none), through the camera's `jitter` if it has one: then
// `jittered` 1, else the offset 0 and `jittered` 0.
contracts::FrameConstants frame_constants(
    const FrameInputs& inputs, Extent at, std::uint32_t accumulated_frames,
    std::optional<sampling::CameraJitter> jitter);

// Contract 17's, for the features at `size`: whether there is a frame
// before, and its camera framed for `size` (camera/thin_lens.h,
// shader_form). For the features at the light size and the denoiser's
// full-size albedo alike. Precondition: the frame before's camera can be
// framed (refusals.h, previous_camera_refusal), as shader_form's.
contracts::FeatureConstants feature_constants(
    const std::optional<PreviousFrame>& previous, Extent size);

// Contract 15's reuse constants: whether there is a frame before to merge,
// and its camera framed for the light's size. Precondition as above.
contracts::PathReuseConstants path_reuse_constants(
    const std::optional<contracts::Camera>& previous_camera, Extent light);

// What contract 18's constants are made from, beyond the camera.
struct DenoiseFrame {
  Extent light;   // the light's size, the network's input
  Extent padded;  // the U-Net's grid (core/denoise/unet.h)
  Extent output;  // the image written
  // the U-Net's last width, the head's inputs a pixel
  std::uint32_t features = 0;
  bool history = false;           // last frame's output is this frame's history
  sampling::CameraJitter jitter;  // the frame's, the light's rays took
};

// Contract 18's: `camera`'s view axes (camera/thin_lens.h, view_axes), and
// `frame`'s sizes, history and jitter. Precondition: `camera` can be
// framed, as view_axes's.
contracts::DenoiseConstants denoise_constants(const contracts::Camera& camera,
                                              const DenoiseFrame& frame);

// Contract 20's: whether the fill's history holds this history's frame
// before.
contracts::CheckerboardConstants checkerboard_constants(bool history);

// The grid ReSTIR PT's one-thread-a-pixel kernels run over at the light's
// size (contract 20): the traced lattice's, half the width rounded up,
// when the light is traced checkerboarded, else the light's.
Extent traced_grid(Extent light, bool checkerboard);

}  // namespace serenity::frame
