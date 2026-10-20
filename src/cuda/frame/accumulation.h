#pragma once

#include <cstdint>
#include <memory>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "cuda/device/device.h"
#include "cuda/device/images.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (history across frames: contract 6, in its first form),
// on the GPU.
//
// The image a converging pass averages its frames into, Metal's
// Accumulation (metal/frame/accumulation.h, which says what each pixel
// holds and how the pass folds a frame in): RGBA32Float at the frame's
// size, pitched linear memory (cuda/device/images.h). Which frames it holds,
// and whether a frame may join it, is the core's rule (core/frame/
// history.h), kept by a frame::History this object owns; the backend
// decides nothing about it (principle 10). Its side is the GPU's: make the
// image at the size the rule says.
//
// prepare() is called once a frame, before the frame's submission begins.
// It asks the rule whether the frame joins and returns the count of frames
// the image holds before it; where the rule says the image is made anew (a
// first frame, or a start over at another size) it makes it, first
// finishing the frames in flight, which may still read the old one (a
// resize is rare: not on any budget). A frame the rule refuses is refused
// by CudaError with the rule's reason, and nothing changes: the rule is
// asked on a copy of the history, kept once the image exists (E.4).
//
// The order between frames is the stream's: one stream runs every pass in
// order, so the pass reading the image reads what the frame before wrote.
// The path pass is the only converging pass this backend builds today;
// ReSTIR PT's shade joins it in Phase 4.
class Accumulation {
 public:
  Accumulation(const Device& device, Submission& submission);

  Accumulation(const Accumulation&) = delete;
  Accumulation& operator=(const Accumulation&) = delete;
  Accumulation(Accumulation&&) = delete;
  Accumulation& operator=(Accumulation&&) = delete;
  ~Accumulation() = default;

  // The count of frames the image holds before frame `inputs`, at `size`,
  // of a scene that changes or not; see above. Throws CudaError if the
  // core's rule refuses the frame or the image cannot be made.
  std::uint32_t prepare(const frame::FrameInputs& inputs, frame::Extent size,
                        frame::SceneMotion motion);

  // The image, valid until the next prepare(); null before the first.
  const PitchedImage* image() const noexcept { return image_.get(); }

 private:
  const Device& device_;
  Submission& submission_;
  std::unique_ptr<PitchedImage> image_;
  frame::History history_;  // the core's rule, and what the image holds
};

}  // namespace serenity::cuda
