#include "metal/frame/offline_frames.h"

#include <memory>
#include <optional>
#include <vector>

namespace serenity::metal {

OfflineFrames::OfflineFrames(const frame::Schedule& schedule,
                             const scene::SceneDescription* scene,
                             frame::Extent size)
    : submission_(device_),
      target_(device_, submission_, size),
      renderer_(device_, submission_, schedule, scene),
      first_sample_(submission_.next_sequence()) {
  settled_.reserve(frames_in_flight);
}

void OfflineFrames::keep(const Completed& completed) {
  if (completed.sequence >= first_sample_) {
    settled_.push_back(SettledSample{completed.sequence, completed.gpu_start,
                                     completed.gpu_end});
  }
}

std::uint64_t OfflineFrames::render(const frame::FrameInputs& inputs) {
  const WindowFrame frame =
      render_offscreen_frame(submission_, target_, renderer_, inputs);
  if (frame.settled) {
    keep(*frame.settled);
  }
  return frame.sequence;
}

void OfflineFrames::wait(std::uint64_t sequence) {
  if (const std::optional<Completed> settled =
          submission_.wait_until_complete(sequence)) {
    keep(*settled);
  }
}

std::uint64_t OfflineFrames::non_finite_samples() const {
  return renderer_.non_finite_samples();
}

std::uint64_t OfflineFrames::mnee_lost_requests() const {
  return renderer_.mnee_lost_requests();
}

std::size_t OfflineFrames::displayed_bytes() const {
  return target_.rgba_size();
}

void OfflineFrames::read_displayed(std::span<std::uint8_t> out) {
  target_.read_rgba(out);
}

void OfflineFrames::read_accumulated(std::span<float> out) {
  renderer_.read_accumulated(out);
}

presentation::OfflineRenderer::Features OfflineFrames::read_features() {
  return renderer_.read_features();
}

void OfflineFrames::finish() {
  for (const Completed& settled : submission_.finish()) {
    keep(settled);
  }
}

void OfflineFrames::take_settled(std::vector<SettledSample>& into) {
  into.insert(into.end(), settled_.begin(), settled_.end());
  settled_.clear();
}

}  // namespace serenity::metal

namespace serenity::presentation {

std::unique_ptr<OfflineRenderer> make_offline_renderer(
    const frame::Schedule& schedule, const scene::SceneDescription* scene,
    frame::Extent size) {
  return std::make_unique<metal::OfflineFrames>(schedule, scene, size);
}

}  // namespace serenity::presentation
