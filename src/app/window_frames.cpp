#include "app/window_frames.h"

#include <chrono>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

#include "app/options.h"

namespace serenity::app {

namespace {

// How often the title's frame times are refreshed.
constexpr frame::Seconds title_period{1.0};

std::string title(const presentation::FrameReport& report,
                  const measurement::Summary& summary) {
  using Milliseconds = std::chrono::duration<double, std::milli>;
  std::string result = std::format(
      "Serenity  |  {} x {}  |  GPU {:.2f} ms mean, {:.2f} to {:.2f}, over {} "
      "frames (live window, not a benchmark)",
      report.size.width, report.size.height, Milliseconds{summary.mean}.count(),
      Milliseconds{summary.shortest}.count(),
      Milliseconds{summary.longest}.count(), summary.frames);
  if (report.non_finite_samples != 0) {
    result += std::format("  |  {} SAMPLES NOT FINITE (a bug)",
                          report.non_finite_samples);
  }
  if (report.mnee_lost_requests != 0) {
    result += std::format("  |  {} MNEE REQUESTS LOST (a bug)",
                          report.mnee_lost_requests);
  }
  return result;
}

}  // namespace

WindowFrames::WindowFrames(Window& window, const scene::SceneDescription* scene,
                           double scale)
    : window_(window),
      scene_(scene),
      scale_(scale),
      frame_times_(title_period),
      // When the image starts over is the core's plan (core/frame/history.h).
      history_(scene && scene::changes(*scene) ? frame::SceneMotion::changing
                                               : frame::SceneMotion::still) {}

bool WindowFrames::events(frame::Seconds longest) {
  return !window_.get().wait(longest).quit;
}

frame::Extent WindowFrames::size() {
  const frame::Extent pixels = window_.get().size_in_pixels();
  if (pixels.width == 0 || pixels.height == 0) {
    return frame::Extent{0, 0};
  }
  return render_size(pixels, scale_);
}

frame::FrameInputs WindowFrames::inputs(frame::Seconds shown_at,
                                        frame::View view) {
  pending_ = frame::FrameInputs{
      .time = clock_.at(shown_at),
      .index = index_,
      .accumulated_since = history_.since(index_, view),
      .camera = scene_ ? std::optional{scene_->camera} : std::nullopt,
      .reused_since = reuse_.since(index_, view),
      .previous = previous_,
  };
  return *pending_;
}

void WindowFrames::shown(const presentation::FrameReport& report) {
  if (!pending_) {
    throw std::logic_error(
        "WindowFrames::shown: no frame was given by inputs() to show");
  }
  const frame::FrameInputs shown_frame = *pending_;
  pending_.reset();
  ++index_;
  if (shown_frame.camera) {
    previous_ = frame::PreviousFrame{.time = shown_frame.time,
                                     .camera = *shown_frame.camera};
  }
  if (report.settled) {
    frame_times_.add(report.settled->gpu_start, report.settled->gpu_end);
  }
  if (const auto summary = frame_times_.take(shown_frame.time)) {
    const std::string line = title(report, *summary);
    window_.get().set_title(line);
    std::cout << line << '\n' << std::flush;
  }
}

}  // namespace serenity::app
