#include "cuda/frame/accumulation.h"

#include "cuda/device/error.h"

namespace serenity::cuda {

Accumulation::Accumulation(const Device& device, Submission& submission)
    : device_(device), submission_(submission) {}

std::uint32_t Accumulation::prepare(const frame::FrameInputs& inputs,
                                    frame::Extent size,
                                    frame::SceneMotion motion) {
  // The rule is asked on a copy, kept only once the image it describes
  // exists (E.4).
  frame::History next = history_;
  const frame::Joined joined = [&] {
    try {
      return next.join(inputs, size, motion);
    } catch (const frame::HistoryError& refused) {
      throw CudaError(refused.what());
    }
  }();
  if (!joined.remade) {
    history_ = next;
    return joined.held;
  }
  // Starting over at a new size: the image's contents are not read (the
  // pass treats every pixel as empty), so only its size matters. Frames in
  // flight may still read the old image: finished first, so a failure
  // among them is reported. Until the new one is made the history holds
  // nothing.
  if (image_) {
    submission_.finish();
    image_.reset();
    history_ = frame::History{};
  }
  image_ = std::make_unique<PitchedImage>(device_, submission_, size);
  history_ = next;
  return joined.held;
}

}  // namespace serenity::cuda
