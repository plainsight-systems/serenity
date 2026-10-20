#include "cuda/frame/path_reuse_images.h"

#include <string>

#include "core/passes/restir_pt_arrays.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

PathReuseImages::PathReuseImages(Submission& submission, Spatial spatial,
                                 Temporal temporal, std::uint32_t lights)
    : submission_(submission),
      spatial_(spatial),
      temporal_(temporal),
      counter_(sizeof(std::uint32_t)),
      light_tiles_(passes::light_tiles_bytes()) {
  zero_now(counter_.address(), sizeof(std::uint32_t), "PathReuseImages");
  const std::size_t copies = temporal == Temporal::merged ? 2 : 1;
  for (std::size_t h = 0; h < copies; ++h) {
    light_power_.at(h) = DeviceBuffer(passes::light_power_bytes(lights));
  }
}

std::uint64_t PathReuseImages::prepare(const frame::FrameInputs& inputs,
                                       frame::Extent size) {
  const bool temporal = temporal_ == Temporal::merged;
  // The history joined on a copy, kept only once the arrays that hold the
  // frame exist: a throw below leaves it claiming nothing they never held
  // (E.4), as Accumulation's.
  frame::ReuseHistory joined = history_;
  std::uint64_t held = 0;
  if (temporal) {
    try {
      held = joined.join(inputs, size).held;
    } catch (const frame::HistoryError& refused) {
      throw CudaError(std::string("ReSTIR PT's reuse history: ") +
                      refused.what());
    }
  }
  if (size == size_) {
    history_ = joined;
    return held;
  }
  // A new size starts the history over (the rule refuses one that does
  // not), so nothing of the old arrays is read again; frames in flight may
  // still read them, so finished first.
  if (reservoirs_.address() != 0) {
    submission_.finish();
  }
  size_ = {};
  const passes::PathReuseArrays sizes =
      passes::path_reuse_arrays(size, spatial_ == Spatial::merged, temporal);
  const auto make = [](std::size_t bytes) {
    return bytes == 0 ? DeviceBuffer() : DeviceBuffer(bytes);
  };
  reservoirs_ = make(sizes.reservoirs);
  for (std::size_t h = 0; h < primaries_.size(); ++h) {
    primaries_.at(h) =
        h < sizes.copies ? make(sizes.primaries) : DeviceBuffer();
    finals_.at(h) = h < sizes.copies ? make(sizes.finals) : DeviceBuffer();
  }
  light_ = make(sizes.light);
  shifts_ = make(sizes.shifts);
  replays_ = make(sizes.replays);
  previous_positions_ = make(sizes.previous_positions);
  previous_pixels_ = make(sizes.previous_pixels);
  duplication_ = make(sizes.duplication);
  size_ = size;
  history_ = joined;
  return held;
}

PathReuseImages::Addresses PathReuseImages::addresses(
    std::uint64_t index) const noexcept {
  const bool temporal = temporal_ == Temporal::merged;
  const std::size_t now = passes::this_frame_copy(index, temporal);
  const std::size_t then = 1 - now;
  Addresses out;
  out.reservoirs = reservoirs_.address();
  out.primaries = primaries_.at(now).address();
  out.light = light_.address();
  out.shifts = shifts_.address();
  out.replays = replays_.address();
  out.counter = counter_.address();
  out.light_tiles = light_tiles_.address();
  out.light_power = light_power_.at(now).address();
  if (temporal) {
    out.last_primaries = primaries_.at(then).address();
    out.last_light_power = light_power_.at(then).address();
    out.finals = finals_.at(now).address();
    out.last_finals = finals_.at(then).address();
    out.previous_positions = previous_positions_.address();
    out.previous_pixels = previous_pixels_.address();
    out.duplication = duplication_.address();
  }
  return out;
}

}  // namespace serenity::cuda
