#pragma once

#include <cuda.h>

#include <array>
#include <cstdint>
#include <vector>

#include "core/frame/frame_inputs.h"
#include "cuda/device/device.h"
#include "cuda/device/frame_sequence.h"

namespace serenity::cuda {

// A frame begun: the stream to record on, its first part's (the light
// stream), its slot, its sequence and its parts (below, "Two parts").
struct FrameSlot {
  CUstream stream = nullptr;
  std::uint32_t slot = 0;
  std::uint64_t sequence = 0;
  Parts parts = Parts::one;
};

// A frame settled: its sequence, and when its work began and ended on the
// GPU's timeline, from the submission's origin (below, "Instants").
struct Completed {
  std::uint64_t sequence = 0;
  frame::Seconds gpu_start{0.0};
  frame::Seconds gpu_end{0.0};
};

// Axis: GPU backend.
//
// How frames reach the GPU, Metal's protocol (metal/device/submission.h)
// on CUDA's stream: begin() a frame, record its work on the stream, commit()
// it; wait_until_complete() before reading back what it wrote; finish() at
// the end of a run. Each frame has a sequence number, from 0, and a slot,
// sequence % frames_in_flight; at most frames_in_flight are committed and
// not yet known complete (GPU.7: the CPU records frame n + 1 while the GPU
// runs frame n, and a slot's resources are written again only once its
// frame has completed). Two streams (below, "Two parts"), each
// non-blocking with respect to the legacy default stream; an event per
// slot, recorded at commit(), marks its frame complete.
//
// Settling. A frame is settled once its event is known complete: by
// begin(), for the frame two submissions back whose slot it reuses; by
// wait_until_complete(), for the one it waits on and every one before; or
// by finish(). The driver reports a fault in a kernel at the next call that
// synchronizes (error.h), so settling throws CudaError for it, and the run
// ends with it, as Metal's settling reports a failed command buffer.
//
// The bookkeeping, which frame takes which slot and which are settled, and
// the misuses refused, is FrameSequence's (frame_sequence.h), apart from the
// driver; this class waits on the events and tells it what it settled.
//
// Two parts (Metal's metal/device/submission.h, "Two parts"; docs/research/
// 2026-10-19-cuda-backend.md, "Phase 5's design"). begin(Parts::two)
// begins a frame whose work is in two halves on two streams: the light
// half on the frame's stream, as a one-part frame's; then hand_off(),
// which records the slot's hand-off event on it and makes the image
// stream wait for that event, and returns the image stream, where the
// image half is recorded. So frame N's image half may run beside frame
// N + 1's light half (proven on the RTX 3090 Ti before this header), and
// each half follows its own half of the frame before on its own stream
// (GPU.7, GPU.8). The frame's completion event is recorded on the stream
// its last work is on, the image stream, so a slot is reused only once its
// image half has completed. A one-part frame after a two-part one waits,
// on the light stream, for that frame's completion (frame_sequence.h).
// What the image half reads of the light half's is ringed by the slot,
// the renderer's (cuda/frame/renderer.h).
//
// Instants. Each frame's GPU instants are two events: one recorded on the
// light stream as the frame begins, and its completion event; each read,
// when the frame settles, as cuEventElapsedTime from an event recorded on
// the light stream when the submission is made. That is the GPU's
// timeline, not host time (TLM.11: the GPU's clock is not the CPU's), and
// what is measured from it subtracts its instants only from each other
// (core/measurement/frame_times.h). The driver gives the span in float
// milliseconds, to about 0.5 us, and past the run's first minutes to
// float's step (0.004 ms at a minute from the origin): a frame's time is
// read to a few microseconds at that age, which the summary's milliseconds
// do not show. Every settled frame's instants are kept until
// take_completed() hands them over, in order, each once; a caller that
// takes them each frame keeps them from growing (MEM.9: the list's
// capacity is made once, for frames_in_flight frames).
//
// Lifetime. Everything a frame uses must outlive the frame's work on the
// GPU: the destructor waits for the stream, and every owner of device
// memory a frame uses is released after it (its renderer's destructor
// waits, as Metal's does). Made once, outside the frame loop; nothing is
// allocated per frame (MEM.9).
//
// Throws CudaError if the stream or its events cannot be made.
class Submission {
 public:
  explicit Submission(const Device& device);

  Submission(const Submission&) = delete;
  Submission& operator=(const Submission&) = delete;
  Submission(Submission&&) = delete;
  Submission& operator=(Submission&&) = delete;
  ~Submission();

  // Begins the next frame, in `parts` parts, settling the one whose slot it
  // reuses; its start event recorded. Throws CudaError if a frame is begun
  // and not committed, or settling fails.
  FrameSlot begin(Parts parts = Parts::one);

  // Hands the frame begun last, in two parts, from its light half to its
  // image half (above); the image stream, to record the image half on.
  // Throws CudaError for a frame of one part, or one already handed off.
  CUstream hand_off();

  // Commits the frame begun last: its completion event recorded after its
  // work, on the image stream for two parts. Throws CudaError for a frame
  // of two parts not handed off.
  void commit();

  // Blocks until frame `sequence`, and every one before it, has completed,
  // settling each; throws CudaError for a fault, or for a sequence not yet
  // committed.
  void wait_until_complete(std::uint64_t sequence);

  // Blocks until every committed frame has completed, settling each; throws
  // CudaError for a fault. Frames may begin again after it.
  void finish();

  // Whether frame `sequence` has completed, without waiting.
  bool has_completed(std::uint64_t sequence) const noexcept;

  // The sequence the next begin() hands out.
  std::uint64_t next_sequence() const noexcept { return frames_.next(); }

  // The stream a two-part frame's image half is recorded on. Work put on it
  // before a frame's hand_off() runs after the image half before and
  // before that frame's image half: a test's delay (tests/cuda/support/
  // hand_off_rig.h).
  CUstream image_stream() const noexcept { return image_; }

  // Appends to `into` the instants of every frame settled since the last
  // call, in order, and forgets them (above, "Instants").
  void take_completed(std::vector<Completed>& into);

  // Blocks until both streams are idle; reports nothing, for destructors.
  void wait_idle() noexcept;

 private:
  void settle(std::uint32_t slot);
  // The span from the origin to `event`, which has completed.
  frame::Seconds since_origin(CUevent event) const;

  CUstream stream_ = nullptr;  // the light stream: a one-part frame's
  CUstream image_ = nullptr;   // a two-part frame's image half
  CUevent origin_ = nullptr;
  // By slot: the frame's start, its completion, its hand-off.
  std::array<CUevent, frames_in_flight> starts_{};
  std::array<CUevent, frames_in_flight> events_{};
  std::array<CUevent, frames_in_flight> hand_offs_{};
  FrameSequence frames_;
  std::vector<Completed> completed_;
};

}  // namespace serenity::cuda
