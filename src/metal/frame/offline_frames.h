#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/presentation/offline_renderer.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace serenity::metal {

// Axis: GPU backend (the headless renderer's side of Metal).
//
// Metal's offline renderer (core/presentation/offline_renderer.h): the
// device, the submission, an offscreen target of the run's size
// (offscreen.h) and the renderer (renderer.h), made once, in that order,
// and released in reverse after the frames in flight (submission.h,
// Lifetime); each of the interface's calls one of theirs. This file also
// defines the factory, make_offline_renderer(), which the headless program
// reaches by linking this backend: the program names no backend.
//
// Each frame is recorded and committed by render_to_offscreen() (renderer.h)
// and waited for by the submission; the reads are the target's and the
// renderer's own, each a readback (GPU.1). The renderer's and the
// target's refusals throw MetalError, a std::exception, with the words they
// had (I.10).
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

}  // namespace serenity::metal
