#pragma once

#include <cstdint>
#include <optional>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"

namespace serenity::presentation {

// Axis: Presentation (contract).
//
// The seam between the window and a backend's presentation
// (logical-overview.md, "The frame", "Pacing"; change-axes.md): a frame
// source, which owns the window's loop and decides when frames come and
// what they are rendered into, and its client, the window, which handles
// its events and says what each frame is. Neither side's types cross it:
// an instant, a size, the frame's inputs, and what a shown frame reports.
//
// The shape is SDL3's main callbacks (SDL_AppEvent, SDL_AppIterate; SDL's
// README-main-functions): the platform that would rather own the loop owns
// it, and the app's code is the same whichever does. Here:
//   - Metal's source (metal/frame/display_frames.h) waits in the client's
//     events while a display link calls frames from the main run loop,
//     as SDL3 drives SDL_AppIterate from a CADisplayLink on iOS;
//   - a Vulkan or Direct3D 12 source would run the loop an engine's game
//     loop runs, Unreal's among them (its engine loop pumps the platform's
//     messages, then renders, and its RHI's present waits on the
//     swapchain): the events polled without waiting, an image acquired,
//     the frame rendered and presented.
// So the client never assumes either: frames may be asked for during
// events() (a run loop calling them while the window waits) or between
// calls (a loop of the source's own), and the client allows both. Each of
// its calls finishes what it does before returning, so a frame called in
// the middle of a wait finds it whole (E.4: valid at every call out).
//
// Both are interfaces of pure virtual functions (I.25, C.121), destroyed
// never through the interface, so their destructors are protected and not
// virtual (C.35). One thread, the window's (CP.1): every call is made on
// the thread that runs the source.

// A rate range in frames a second (Hz): what the window asks of the
// source. A source asks its platform for it (CAFrameRateRange; a
// swapchain's presentation timing) and refuses a range that is not
// positive and in order (minimum <= preferred <= maximum).
struct FrameRate {
  float minimum = 0.0F;
  float maximum = 0.0F;
  float preferred = 0.0F;
};

// An earlier frame's time on the GPU, in host time, once the backend has
// settled it: what the frame times count (core/measurement/frame_times.h).
// Only frames are reported, never start-up work.
struct SettledFrame {
  frame::Seconds gpu_start{0.0};
  frame::Seconds gpu_end{0.0};
};

// What a frame shown reports, for the window's title.
struct FrameReport {
  frame::Extent size;  // the size it was rendered at
  // a frame its submission settled, if one
  std::optional<SettledFrame> settled;
  // samples left out as not finite, so far (a bug)
  std::uint64_t non_finite_samples = 0;
  // caustic requests lost, so far (a bug)
  std::uint64_t mnee_lost_requests = 0;
};

class FrameClient {
 public:
  // Handles the window's events since the last call, waiting for one at
  // most `longest` (0: not at all). Returns whether the run goes on: false
  // once the window is closed. May throw; the source then stops and lets
  // it through (E.2).
  virtual bool events(frame::Seconds longest) = 0;

  // The size the next frame is to be rendered at: the window's pixels at
  // its scale; a side of 0 when the window shows nothing (minimized), and
  // then the source renders none.
  virtual frame::Extent size() = 0;

  // The inputs of the frame the source is about to render, which will be
  // shown at `shown_at` (host time, the source's instant); `view`
  // changed when the size the source renders at changed since the last
  // frame shown. Each call is followed by shown(), once the frame is
  // rendered and presented, unless rendering fails, which ends the run.
  // May throw, as rendering may (below).
  virtual frame::FrameInputs inputs(frame::Seconds shown_at,
                                    frame::View view) = 0;

  // The frame inputs() gave was rendered and presented.
  virtual void shown(const FrameReport& report) = 0;

 protected:
  FrameClient() = default;
  FrameClient(const FrameClient&) = default;
  FrameClient& operator=(const FrameClient&) = default;
  FrameClient(FrameClient&&) = default;
  FrameClient& operator=(FrameClient&&) = default;
  ~FrameClient() = default;
};

class FrameSource {
 public:
  // Runs the window: calls `client`'s events() and, as the source paces
  // them, its frames, until events() returns false. Throws what a frame
  // or the client threw, or the source's own failure, having stopped
  // calling frames first (E.2, E.6: what it started for the run is
  // released on the way out). Called once.
  virtual void run(FrameClient& client) = 0;

 protected:
  FrameSource() = default;
  FrameSource(const FrameSource&) = default;
  FrameSource& operator=(const FrameSource&) = default;
  FrameSource(FrameSource&&) = default;
  FrameSource& operator=(FrameSource&&) = default;
  ~FrameSource() = default;
};

}  // namespace serenity::presentation
