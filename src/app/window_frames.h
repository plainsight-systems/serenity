#pragma once

#include <cstdint>
#include <functional>
#include <optional>

#include "app/clock.h"
#include "app/window.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "core/measurement/frame_times.h"
#include "core/presentation/frame_source.h"
#include "core/scene/scene.h"

namespace serenity::app {

// Axis: Presentation.
//
// The window's side of the frame source (core/presentation/frame_source.h,
// FrameClient): what each frame is, and the window's events, whatever
// paces the frames. It names no backend (make check, rule 6), so the same
// code serves Metal's display link and a swapchain's loop alike; main.cpp
// composes it with a backend's source.
//
// - events(): the window's events, waiting at most as long as the source
//   says (window.h, wait); the run goes on until the window is closed,
//   Escape is pressed or the program is interrupted.
// - size(): the window's pixels at the scale (options.h, render_size);
//   none while the window has no pixels (minimized).
// - inputs(): the frame's time is the instant it will be shown, counted
//   from the first frame's (clock.h), the count of frames shown, the
//   scene's camera, still until the camera moves, and the frame before it
//   (its time and camera, which a graph with the denoiser's features
//   measures motion from; core/frame/frame_inputs.h, previous; none before
//   the first). A graph that converges accumulates from the first frame
//   and starts over as the core's plan says (core/frame/history.h,
//   LiveHistory): when the size changed (`view`); every frame while the
//   scene's shapes move; when the image would hold more than it can. The
//   reuse history likewise (LiveReuse).
// - shown(): the frame counts as shown, so the next is the next index and
//   has it as the frame before; the frame its submission settled joins the
//   frame times, and once a second the title shows them
//   (core/measurement/frame_times.h), with the bug counters if not 0. It is
//   a live window's figure, with the window's own work beside it, not a
//   benchmark, and the title says so (TLM.6). The frames the backend
//   settles at the end come after the last title and are not shown.
//
// inputs() is followed by shown() for the frame it gave (frame_source.h),
// so the frame it gives is held until then (pending_); each call finishes
// before it returns, so a frame called during events() finds the state
// whole (E.4). Throws what the core's plans throw (HistoryError for frames
// out of order, I.5) and what the window throws (WindowError, E.2); and
// std::logic_error for a shown() with no frame given, a source breaking
// the contract (I.6), which counts nothing.
//
// Not performance-sensitive: a handful of values a frame; the title's text
// is made once a second.
class WindowFrames final : public presentation::FrameClient {
 public:
  // `scene`, none for a graph that reads none, and `window` must outlive
  // this object; `scale` is in (0, 1] (options.h).
  WindowFrames(Window& window, const scene::SceneDescription* scene,
               double scale);

  bool events(frame::Seconds longest) override;
  frame::Extent size() override;
  frame::FrameInputs inputs(frame::Seconds shown_at, frame::View view) override;
  void shown(const presentation::FrameReport& report) override;

 private:
  std::reference_wrapper<Window> window_;  // C.12: not a reference member
  const scene::SceneDescription* scene_;
  double scale_;

  PresentationClock clock_;
  measurement::FrameTimes frame_times_;
  std::uint64_t index_ = 0;
  frame::LiveHistory history_;
  frame::LiveReuse reuse_;
  // the last frame shown, none before the first
  std::optional<frame::PreviousFrame> previous_;
  // the frame inputs() gave, until shown()
  std::optional<frame::FrameInputs> pending_;
};

}  // namespace serenity::app
