#pragma once

#include <cuda.h>

#include <cstdint>
#include <span>

#include "cuda/device/frame_upload.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Scene content (on the GPU: how bright each light is).
//
// Each sphere light's glow as a frame lights it (core/animation/glow.h), a
// factor on its radiance, one float a sphere light in the scene's order:
// Metal's LightGlows (metal/scene/light_glows.h, which says what reads
// them). How they reach the GPU a frame is FrameUpload's (cuda/device/
// frame_upload.h). Every copy starts at 1; in a scene where some light
// glows, the core rewrites the glowing lights' factors in the slot's copy
// and the frame uploads it. A scene with no sphere lights keeps one factor,
// unread, since an array must have bytes. Last frame's glows arrive with
// ReSTIR PT (Phase 4).
//
// Throws CudaError if the memory cannot be made.
class LightGlows {
 public:
  LightGlows(Submission& submission, std::uint32_t lights);

  LightGlows(const LightGlows&) = delete;
  LightGlows& operator=(const LightGlows&) = delete;
  LightGlows(LightGlows&&) = delete;
  LightGlows& operator=(LightGlows&&) = delete;
  ~LightGlows() = default;

  // Frame slot `slot`'s factors, to light the frame's lights in, under
  // FrameUpload's rule for writing.
  std::span<float> glows(std::uint32_t slot) {
    // The bytes were made from floats, at least one: a scene with no sphere
    // lights has one, unread, and is given none.
    return array_.view<float>(slot).first(lights_);
  }

  // The frame's factors, from its slot's copy, on its stream.
  void upload(const FrameSlot& frame) { array_.upload(frame); }

  // The factors, as a program reads them.
  CUdeviceptr address() const noexcept { return array_.address(); }

 private:
  std::uint32_t lights_;
  FrameUpload array_;
};

}  // namespace serenity::cuda
