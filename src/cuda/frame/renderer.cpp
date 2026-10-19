#include "cuda/frame/renderer.h"

#include <string>

#include "core/frame/frame_constants.h"
#include "core/frame/refusals.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// A refusal the core gives, thrown as this backend's error: `prefix` and
// the reason. None, nothing.
void refuse(const std::optional<std::string>& reason,
            std::string_view prefix = "Renderer: ") {
  if (reason) {
    throw CudaError(std::string(prefix) + *reason);
  }
}

}  // namespace

std::optional<std::string> unbuilt_pass(const frame::Schedule& schedule) {
  for (const frame::PassKind kind : schedule.passes) {
    if (kind != frame::PassKind::test_pattern) {
      return "the CUDA backend has no " + std::string(name(kind)) +
             " pass yet (docs/research/2026-10-19-cuda-backend.md, its "
             "phases 3 to 5)";
    }
  }
  return std::nullopt;
}

Renderer::Renderer(Submission& submission, const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : submission_(submission) {
  refuse(frame::invalid(schedule));
  refuse(frame::scene_refusal(schedule, scene));
  refuse(unbuilt_pass(schedule));
  for (const frame::PassKind kind : schedule.passes) {
    if (kind == frame::PassKind::test_pattern) {
      test_pattern_.emplace();
    }
  }
}

Renderer::~Renderer() { submission_.wait_idle(); }

void Renderer::record(const FrameSlot& begun, const frame::FrameInputs& inputs,
                      const Offscreen& target) {
  refuse(frame::time_refusal(inputs.time, inputs.index, "its time"), "");
  const frame::Extent size = target.size();
  const contracts::FrameConstants constants =
      frame::frame_constants(inputs, size, 0u, std::nullopt);
  if (test_pattern_) {
    test_pattern_->record(begun.stream, constants, target.address());
  }
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
  const FrameSlot begun = submission.begin();
  renderer.record(begun, inputs, target);
  submission.commit();
  return begun.sequence;
}

}  // namespace serenity::cuda
