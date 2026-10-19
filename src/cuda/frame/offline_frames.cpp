#include "cuda/frame/offline_frames.h"

#include <memory>

#include "cuda/device/error.h"

namespace serenity::cuda {

OfflineFrames::OfflineFrames(const frame::Schedule& schedule,
                             const scene::SceneDescription* scene,
                             frame::Extent size)
    : submission_(device_),
      target_(submission_, size),
      renderer_(submission_, schedule, scene) {}

std::uint64_t OfflineFrames::render(const frame::FrameInputs& inputs) {
  return render_to_offscreen(submission_, target_, renderer_, inputs);
}

void OfflineFrames::wait(std::uint64_t sequence) {
  submission_.wait_until_complete(sequence);
}

// No pass the backend builds today counts either (renderer.h).
std::uint64_t OfflineFrames::non_finite_samples() const { return 0; }
std::uint64_t OfflineFrames::mnee_lost_requests() const { return 0; }

std::size_t OfflineFrames::displayed_bytes() const {
  return target_.rgba_size();
}

void OfflineFrames::read_displayed(std::span<std::uint8_t> out) {
  target_.read_rgba(out);
}

void OfflineFrames::read_accumulated(std::span<float> /*out*/) {
  throw CudaError(
      "OfflineFrames::read_accumulated: no pass the CUDA backend builds yet "
      "keeps an accumulated image");
}

presentation::OfflineRenderer::Features OfflineFrames::read_features() {
  throw CudaError(
      "OfflineFrames::read_features: the CUDA backend has no features pass "
      "yet");
}

void OfflineFrames::finish() { submission_.finish(); }

}  // namespace serenity::cuda

namespace serenity::presentation {

std::unique_ptr<OfflineRenderer> make_offline_renderer(
    const frame::Schedule& schedule, const scene::SceneDescription* scene,
    frame::Extent size) {
  return std::make_unique<cuda::OfflineFrames>(schedule, scene, size);
}

}  // namespace serenity::presentation
