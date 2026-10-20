#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/presentation/offline_renderer.h"
#include "core/scene/scene.h"
#include "cuda/device/device.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/submission.h"
#include "cuda/frame/renderer.h"

namespace serenity::cuda {

// Axis: GPU backend (the headless renderer's side of CUDA).
//
// CUDA's offline renderer (core/presentation/offline_renderer.h), as
// Metal's (metal/frame/offline_frames.h): the device, the submission, an
// offscreen target of the run's size and the renderer, made once, in that
// order, released in reverse after the frames in flight; and the factory,
// make_offline_renderer(), which the headless program reaches by linking
// this backend. What the backend does not render yet it refuses by name:
// no pass built today counts lost MNEE requests (0, which is what it
// counted) or writes features, so reading them back throws CudaError.
class OfflineFrames final : public presentation::OfflineRenderer {
 public:
  OfflineFrames(const frame::Schedule& schedule,
                const scene::SceneDescription* scene, frame::Extent size);

  OfflineFrames(const OfflineFrames&) = delete;
  OfflineFrames& operator=(const OfflineFrames&) = delete;
  OfflineFrames(OfflineFrames&&) = delete;
  OfflineFrames& operator=(OfflineFrames&&) = delete;
  ~OfflineFrames() override = default;

  std::uint64_t render(const frame::FrameInputs& inputs) override;
  void wait(std::uint64_t sequence) override;
  std::uint64_t non_finite_samples() const override;
  std::uint64_t mnee_lost_requests() const override;
  std::size_t displayed_bytes() const override;
  void read_displayed(std::span<std::uint8_t> out) override;
  void read_accumulated(std::span<float> out) override;
  Features read_features() override;
  void finish() override;

 private:
  Device device_;
  Submission submission_;
  Offscreen target_;
  Renderer renderer_;
};

}  // namespace serenity::cuda
