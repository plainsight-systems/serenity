#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include "core/frame/extent.h"
#include "core/passes/tone_map.h"
#include "cuda/device/images.h"
#include "cuda/device/submission.h"

namespace serenity::cuda {

// Axis: Frame graph (images a frame's passes share), on the GPU.
//
// The tone map's bloom pyramid (core/passes/tone_map.h, step 2), Metal's
// (metal/frame/frame_images.h): passes::bloom_levels images of four half
// floats a pixel (cuda/device/images.h, LevelImage), level 0 half the
// frame's size and each after it half the one before, at the core's sizes
// (core/passes/bloom.h). Every pixel of every level is written each frame
// before it is read (the tone map's down steps), so a level's contents
// never cross frames, and one pyramid serves every frame: the stream runs
// one frame's tone map after the last's.
//
// prepare() is called once a frame, before the frame's submission begins,
// with the frame's size: it makes the pyramid at its first size and anew
// at another, first finishing the frames in flight, which may still read
// the old one (a resize is rare: not on any budget).
//
// Throws CudaError if an image cannot be made; level() throws CudaError
// before the first prepare() or for a level past the last.
class FrameImages {
 public:
  explicit FrameImages(Submission& submission);

  FrameImages(const FrameImages&) = delete;
  FrameImages& operator=(const FrameImages&) = delete;
  FrameImages(FrameImages&&) = delete;
  FrameImages& operator=(FrameImages&&) = delete;
  ~FrameImages() = default;

  void prepare(frame::Extent size);

  const LevelImage& level(std::uint32_t level) const;

 private:
  Submission& submission_;
  std::optional<frame::Extent> size_;
  std::array<std::unique_ptr<LevelImage>, passes::bloom_levels> pyramid_;
};

}  // namespace serenity::cuda
