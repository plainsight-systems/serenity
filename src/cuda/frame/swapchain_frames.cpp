#include "cuda/frame/swapchain_frames.h"

#include <chrono>
#include <optional>
#include <string>

#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// How long the client's events are waited for while the window shows
// nothing (a minimized window): long enough not to spin, short enough that
// a restored window is seen at once (Metal's display_frames.cpp waits as
// long between its link's frames).
constexpr frame::Seconds event_wait = std::chrono::milliseconds{100};

// The host's steady clock, as the client's instants (frame_source.h).
frame::Seconds now() {
  return std::chrono::duration_cast<frame::Seconds>(
      std::chrono::steady_clock::now().time_since_epoch());
}

presentation::FrameRate checked(presentation::FrameRate rate) {
  if (const std::optional<std::string> why =
          presentation::frame_rate_refusal(rate)) {
    throw CudaError("SwapchainFrames: " + *why);
  }
  return rate;
}

}  // namespace

SwapchainFrames::SwapchainFrames(Submission& submission, Presenter& presenter,
                                 Renderer& renderer,
                                 presentation::FrameRate rate)
    : submission_(submission),
      presenter_(presenter),
      renderer_(renderer),
      rate_(checked(rate)) {
  settled_.reserve(frames_in_flight + 1);
}

void SwapchainFrames::run(presentation::FrameClient& client) {
  first_frame_ = submission_.next_sequence();
  view_changed_ = false;
  settled_.clear();
  reported_ = 0;
  bool rendered = true;
  // The frames' work finished however the loop ends (E.6): a failure in
  // the last frames is then the submission's to report, after the loop's.
  struct Finish {
    Submission& submission;
    explicit Finish(Submission& s) : submission(s) {}
    Finish(const Finish&) = delete;
    Finish& operator=(const Finish&) = delete;
    Finish(Finish&&) = delete;
    Finish& operator=(Finish&&) = delete;
    ~Finish() { submission.wait_idle(); }
  };
  const Finish finish{submission_};
  while (client.events(rendered ? frame::Seconds{0.0} : event_wait)) {
    rendered = render(client);
  }
}

bool SwapchainFrames::render(presentation::FrameClient& client) {
  const frame::Extent wanted = client.size();
  if (wanted.width == 0 || wanted.height == 0) {
    return false;
  }
  if (!(wanted == presenter_.size())) {
    view_changed_ = true;
  }
  presenter_.resize(wanted);
  const frame::Seconds before = now();
  const auto slot = static_cast<std::uint32_t>(submission_.next_sequence() %
                                               frames_in_flight);
  const std::optional<WindowImage> image = presenter_.acquire(slot);
  if (!image) {
    return true;  // remade for the next frame, which comes at once
  }
  const frame::Seconds shown_at =
      before + frame::Seconds{1.0 / static_cast<double>(rate_.preferred)};
  const frame::FrameInputs inputs = client.inputs(
      shown_at, view_changed_ ? frame::View::changed : frame::View::same);
  (void)render_to_window(submission_, presenter_, renderer_, inputs, *image);
  view_changed_ = false;

  // The frames settled so far, the run's alone, one a report (above).
  if (reported_ == settled_.size()) {
    settled_.clear();
    reported_ = 0;
  }
  submission_.take_completed(settled_);
  while (reported_ < settled_.size() &&
         settled_[reported_].sequence < first_frame_) {
    ++reported_;  // start-up work, no frame
  }
  presentation::FrameReport report{
      .size = presenter_.size(),
      .settled = std::nullopt,
      .non_finite_samples = renderer_.non_finite_samples(),
      .mnee_lost_requests = renderer_.mnee_lost_requests(),
  };
  if (reported_ < settled_.size()) {
    const Completed& frame = settled_[reported_++];
    report.settled = presentation::SettledFrame{frame.gpu_start, frame.gpu_end};
  }
  client.shown(report);
  return true;
}

}  // namespace serenity::cuda
