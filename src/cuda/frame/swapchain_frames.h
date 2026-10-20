#pragma once

#include <cstdint>
#include <vector>

#include "core/presentation/frame_source.h"
#include "cuda/device/submission.h"
#include "cuda/frame/renderer.h"
#include "cuda/present/presenter.h"

namespace serenity::cuda {

// Axis: Frame graph.
//
// CUDA's frame source (core/presentation/frame_source.h), Metal's
// DisplayFrames' counterpart (metal/frame/display_frames.h): the loop an
// engine's game loop runs, as that header has a swapchain's source run it
// (docs/research/2026-10-20-cuda-window.md). run(), on the main thread,
// until the client's events() returns false:
//
//   1. the client's events, polled without waiting while frames come,
//      waited for at most event_wait while the window shows nothing;
//   2. the size the client wants; none rendered at a side of 0 (a
//      minimized window); the presenter follows it (presenter.h, resize),
//      a change told to the next frame rendered as View::changed;
//   3. a swapchain image acquired for the slot the next frame takes, which
//      paces the loop at the display's rate (FIFO; swapchain_choice.h);
//      none if the swapchain was out of date, remade for the next frame;
//   4. the client's inputs, for the instant the frame is expected on
//      screen: the host's steady clock before the acquire, and one frame of
//      the rate asked for. Vulkan reports no presentation time without an
//      extension NVIDIA's X11 driver lacks, so this estimate is the
//      instant: what it decides is the animation's time, which advances by
//      the frames' spacing as on the Mac;
//   5. the frame rendered into the slot's target and presented
//      (render_to_window, renderer.h);
//   6. the client's shown(), with the size, a frame its submission settled
//      (its GPU instants, CUDA's timeline: submission.h, "Instants"), and
//      the renderer's bug counters.
// Which submissions are frames is this source's to know: those from the
// run's start on, start-up work having been submitted before. A frame
// settles as a later one begins, one a frame in a steady run; any more,
// as a wait settles several, are reported one a shown() in order, kept in
// a list made once (MEM.9). A frame that throws ends the run with it, the
// submission's frames finished first (E.2, E.6).
//
// The rate asked for is checked (presentation::frame_rate_refusal) and
// refused by CudaError when the source is made; FIFO presents at the
// display's own rate, which the rate's preferred frames a second are
// taken to be, for the instants above.
//
// Threads (CP.1): the main thread only. The submission, presenter and
// renderer outlive it; the client is held only while run() runs. Copying
// and moving are deleted (C.21).
class SwapchainFrames final : public presentation::FrameSource {
 public:
  SwapchainFrames(Submission& submission, Presenter& presenter,
                  Renderer& renderer, presentation::FrameRate rate);

  SwapchainFrames(const SwapchainFrames&) = delete;
  SwapchainFrames& operator=(const SwapchainFrames&) = delete;
  SwapchainFrames(SwapchainFrames&&) = delete;
  SwapchainFrames& operator=(SwapchainFrames&&) = delete;
  ~SwapchainFrames() = default;

  void run(presentation::FrameClient& client) override;

 private:
  // Steps 2 to 6 above; whether a frame was rendered.
  bool render(presentation::FrameClient& client);

  Submission& submission_;
  Presenter& presenter_;
  Renderer& renderer_;
  presentation::FrameRate rate_;

  // During run() only.
  std::uint64_t first_frame_ = 0;
  bool view_changed_ = false;
  std::vector<Completed> settled_;  // taken from the submission, in order
  std::size_t reported_ = 0;        // of settled_, those shown
};

}  // namespace serenity::cuda
