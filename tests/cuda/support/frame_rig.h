#pragma once

// CUDA's frames for a test that drives renderers itself (tests/gpu/
// features_test.cpp): one device and submission, the targets and renderers
// a test asks for, and a frame rendered into a target and waited for.
// Metal's is tests/gpu/support/frame_rig.h, under the same names and
// interface, so such a test is one source for both backends. A fixture:
// what it makes refers to its device and submission (C.8).

#include <memory>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "cuda/device/device.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/submission.h"
#include "cuda/frame/renderer.h"

namespace serenity::tests {

class FrameRig {
 public:
  using Target = cuda::Offscreen;
  using Renderer = cuda::Renderer;

  FrameRig() = default;
  FrameRig(const FrameRig&) = delete;
  FrameRig& operator=(const FrameRig&) = delete;
  FrameRig(FrameRig&&) = delete;
  FrameRig& operator=(FrameRig&&) = delete;
  ~FrameRig() = default;

  std::unique_ptr<Target> target(frame::Extent size) {
    return std::make_unique<Target>(submission_, size);
  }
  std::unique_ptr<Renderer> renderer(const frame::Schedule& schedule,
                                     const scene::SceneDescription* scene) {
    return std::make_unique<Renderer>(device_, submission_, schedule, scene);
  }

  // Frame `inputs`, recorded by `renderer` into `target`, committed and
  // waited for.
  void render(Target& target, Renderer& renderer,
              const frame::FrameInputs& inputs) {
    (void)submission_.wait_until_complete(
        cuda::render_to_offscreen(submission_, target, renderer, inputs));
  }

 private:
  cuda::Device device_;
  cuda::Submission submission_{device_};
};

}  // namespace serenity::tests
