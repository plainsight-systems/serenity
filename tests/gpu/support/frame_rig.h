#pragma once

// Metal's frames for a test that drives renderers itself (tests/gpu/
// features_test.cpp): one device and submission, the targets and renderers
// a test asks for, and a frame rendered into a target and waited for.
// CUDA's is tests/cuda/support/frame_rig.h, under the same names and
// interface, so such a test is one source for both backends. A fixture:
// what it makes refers to its device and submission (C.8).

#include <memory>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace serenity::tests {

class FrameRig {
 public:
  using Target = metal::Offscreen;
  using Renderer = metal::Renderer;

  FrameRig() = default;
  FrameRig(const FrameRig&) = delete;
  FrameRig& operator=(const FrameRig&) = delete;
  FrameRig(FrameRig&&) = delete;
  FrameRig& operator=(FrameRig&&) = delete;
  ~FrameRig() = default;

  std::unique_ptr<Target> target(frame::Extent size) {
    return std::make_unique<Target>(device_, submission_, size);
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
        metal::render_to_offscreen(submission_, target, renderer, inputs));
  }

 private:
  metal::Device device_;
  metal::Submission submission_{device_};
};

}  // namespace serenity::tests
