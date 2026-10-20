#pragma once

// A frame rendered whole on CUDA and read back, and a renderer whose
// frames accumulate (ES.3, F.10): the backend's part of the GPU tests'
// rendering (tests/gpu/support/rendering.h, which includes it by the
// include path), Metal's (tests/gpu/support/accumulating_rig.h) under the
// same names and interface.

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "cuda/device/device.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/submission.h"
#include "cuda/frame/renderer.h"

namespace serenity::tests {

// What `target` holds, read back.
inline std::vector<std::uint8_t> read_back(const cuda::Offscreen& target) {
  std::vector<std::uint8_t> rgba(target.rgba_size());
  target.read_rgba(rgba);
  return rgba;
}

// A renderer of `schedule` over a scene, at a size, rendering frames that
// accumulate from frame 0, each at time 0, its reuse history kept from
// frame 0 too. A fixture: the renderer reads the scene it holds, so it is
// neither copied nor moved (C.8).
class AccumulatingRig {
 public:
  AccumulatingRig(std::string_view text, const frame::Schedule& schedule,
                  frame::Extent size)
      : description_(scene::parse(text, "test scene")),
        size_(size),
        renderer_(device_, submission_, schedule, &description_) {}
  AccumulatingRig(const AccumulatingRig&) = delete;
  AccumulatingRig& operator=(const AccumulatingRig&) = delete;
  AccumulatingRig(AccumulatingRig&&) = delete;
  AccumulatingRig& operator=(AccumulatingRig&&) = delete;
  ~AccumulatingRig() = default;

  // Renders `frames` more frames and returns the last, as displayed, with
  // every sample of them finite.
  std::vector<std::uint8_t> render(std::uint64_t frames) {
    std::uint64_t sequence = 0;
    for (std::uint64_t i = 0; i < frames; ++i, ++next_) {
      sequence = cuda::render_to_offscreen(
          submission_, target_, renderer_,
          frame::FrameInputs{.time = frame::Seconds(0.0),
                             .index = next_,
                             .accumulated_since = 0,
                             .camera = description_.camera});
    }
    (void)submission_.wait_until_complete(sequence);
    CHECK(renderer_.non_finite_samples() == 0);
    return read_back(target_);
  }

  // The accumulated image of the frames rendered so far, RGBA floats
  // (Renderer::read_accumulated).
  std::vector<float> accumulated() {
    std::vector<float> floats(std::size_t{size_.width} * size_.height * 4);
    renderer_.read_accumulated(floats);
    return floats;
  }

  // Renders `frames` more frames and returns the accumulated image.
  std::vector<float> render_accumulated(std::uint64_t frames) {
    (void)render(frames);
    return accumulated();
  }

  const scene::SceneDescription& description() const noexcept {
    return description_;
  }
  frame::Extent size() const noexcept { return size_; }
  cuda::Renderer& renderer() noexcept { return renderer_; }

 private:
  scene::SceneDescription description_;
  frame::Extent size_;
  cuda::Device device_;
  cuda::Submission submission_{device_};
  cuda::Offscreen target_{submission_, size_};
  cuda::Renderer renderer_;
  std::uint64_t next_ = 0;
};

}  // namespace serenity::tests
