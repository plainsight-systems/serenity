#pragma once

#include <cuda.h>

#include <array>
#include <cstdint>
#include <optional>

#include "cuda/device/device.h"

namespace serenity::cuda {

// Axis: GPU backend.
//
// How frames reach the GPU, Metal's protocol (metal/device/submission.h)
// on CUDA's stream: begin() a frame, record its work on the stream, commit()
// it; wait_until_complete() before reading back what it wrote; finish() at
// the end of a run. Each frame has a sequence number, from 0, and a slot,
// sequence % frames_in_flight; at most frames_in_flight are committed and
// not yet known complete (GPU.7: the CPU records frame n + 1 while the GPU
// runs frame n, and a slot's resources are written again only once its
// frame has completed). One stream, non-blocking with respect to the
// legacy default stream; an event per slot, recorded at commit(), marks
// its frame complete.
//
// Settling. A frame is settled once its event is known complete: by
// begin(), for the frame two submissions back whose slot it reuses; by
// wait_until_complete(), for the one it waits on and every one before; or
// by finish(). The driver reports a fault in a kernel at the next call that
// synchronizes (error.h), so settling throws CudaError for it, and the run
// ends with it, as Metal's settling reports a failed command buffer.
//
// Lifetime. Everything a frame uses must outlive the frame's work on the
// GPU: the destructor waits for the stream, and every owner of device
// memory a frame uses is released after it (its renderer's destructor
// waits, as Metal's does). Made once, outside the frame loop; nothing is
// allocated per frame (MEM.9).
//
// Throws CudaError if the stream or its events cannot be made.
inline constexpr std::uint32_t frames_in_flight = 2;

// A frame begun: the stream to record on, its slot and its sequence.
struct FrameSlot {
  CUstream stream = nullptr;
  std::uint32_t slot = 0;
  std::uint64_t sequence = 0;
};

class Submission {
 public:
  explicit Submission(const Device& device);

  Submission(const Submission&) = delete;
  Submission& operator=(const Submission&) = delete;
  Submission(Submission&&) = delete;
  Submission& operator=(Submission&&) = delete;
  ~Submission();

  // Begins the next frame, settling the one whose slot it reuses. Throws
  // CudaError if a frame is begun and not committed, or settling fails.
  FrameSlot begin();

  // Commits the frame begun last: its event recorded after its work.
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
  std::uint64_t next_sequence() const noexcept { return next_; }

  // Blocks until the stream is idle; reports nothing, for destructors.
  void wait_idle() noexcept;

 private:
  void settle(std::uint32_t slot);

  CUstream stream_ = nullptr;
  std::array<CUevent, frames_in_flight> events_{};
  // Each slot's committed frame not yet settled, if any.
  std::array<std::optional<std::uint64_t>, frames_in_flight> pending_{};
  std::uint64_t next_ = 0;
  std::uint64_t settled_through_ = 0;  // every frame before it is settled
  bool open_ = false;                  // a frame begun and not committed
};

}  // namespace serenity::cuda
