#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstddef>
#include <span>
#include <string_view>

#include "core/frame/extent.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: GPU backend (reading images back, for the headless renderer and the
// tests: GPU.1, never in a frame loop).
//
// One sequence for every readback (ES.3, I.5): a shared buffer made
// resident, a submission of its own, a barrier, the copies of whole images
// into the buffer, the commit and the wait. The barrier waits for every
// dispatch committed before it on the queue (barrierAfterQueueStages,
// dispatch before blit): Metal 4 tracks no hazards, so without it a
// readback recorded while a frame is in flight would copy what that frame
// is still writing. With it, a readback sees every frame committed before
// it, finished, and its caller need not wait for them first.

// An image to copy whole into the readback's buffer: its texture, at its
// size, rows of `row_bytes` from `offset`.
struct ImageCopy {
  MTL::Texture* texture = nullptr;
  std::size_t offset = 0;
  std::size_t row_bytes = 0;
};

// Copies each of `copies`, every one `size`, into a new shared buffer of
// `bytes`, through `submission`, and returns the buffer once the copies have
// completed. Throws MetalError naming `what` if the device makes no buffer
// or the command buffer no encoder, or as Submission's wait does.
NS::SharedPtr<MTL::Buffer> read_back(MTL::Device* device,
                                     Submission& submission, frame::Extent size,
                                     std::size_t bytes,
                                     std::span<const ImageCopy> copies,
                                     std::string_view what);

}  // namespace serenity::metal
