#pragma once

#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (history across frames: contract 6, in its first form).
//
// The image a converging pass averages its frames into: the mean radiance of
// frames accumulated_since .. index - 1 (frame/frame_inputs.h), in linear
// RGBA32Float, at the frame's size. The pass reads it, folds the new frame
// in, and writes it back (metal/passes/path/path.h, by the rule in
// metal/film/accumulate.metal.h); this object says how many frames it
// already holds, and keeps that number true.
//
// prepare() is called once a frame, before the frame's submission begins
// (metal/frame/renderer.h, Renderer::prepare), and returns the count of
// frames the image holds, which the frame joins:
//
//   - 0, starting over, when the inputs say so (accumulated_since == index),
//     after (re)making the image if the size changed;
//   - index - accumulated_since, when the image holds exactly the frames
//     accumulated_since .. index - 1 at this size;
//   - otherwise it throws Error: a frame skipped, a size changed or an
//     accumulated_since moved without starting over. The image would then
//     hold something other than what the frame claims to show, and a frame
//     that lies about its image is refused, not shown (principle 1). So is a
//     count past 2^32 - 1, which the frame constants cannot carry (over two
//     years of frames at 60 a second).
//
// The image is the GPU's alone, in private storage, made resident through
// the submission. Remaking it for a new size first drains the submission,
// which is why prepare() comes before a submission begins: frames in flight
// may still read the old image, so it leaves the residency set and is
// released only once they have completed. A resize is rare and the wait is a
// frame or two, so it is not on any budget.
//
// Non-finite samples. The image keeps a counter, in shared memory, of the
// samples its pass left out for not being finite (accumulate.metal.h), over
// every frame since this object was made, whatever size the image was. non_finite_samples() reads it: exact for
// every frame that has completed (Submission::wait_until_complete, drain or
// finish), and a lower bound while frames are in flight.
//
// The order between frames is the pass's to make: Metal 4 does not track
// hazards, so the pass waits, at its start, for the previous submission's
// dispatches to finish writing the image (path.h).
class Accumulation {
public:
    Accumulation(const Device& device, Submission& submission);

    Accumulation(const Accumulation&) = delete;
    Accumulation& operator=(const Accumulation&) = delete;
    Accumulation(Accumulation&&) = delete;
    Accumulation& operator=(Accumulation&&) = delete;
    ~Accumulation() = default;

    // The count of frames the image holds before frame `inputs`, at `size`;
    // see above. Throws Error if the image cannot hold what the inputs claim,
    // or cannot be made.
    std::uint32_t prepare(const frame::FrameInputs& inputs, frame::Extent size);

    // The image, valid until the next prepare(); null before the first.
    MTL::Texture* texture() const { return texture_.get(); }

    // The counter the pass adds to, as a shader binds it.
    MTL::GPUAddress non_finite_counter() const { return counter_->gpuAddress(); }

    // The samples left out for not being finite, see above.
    std::uint32_t non_finite_samples() const;

private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    NS::SharedPtr<MTL::Texture> texture_;
    NS::SharedPtr<MTL::Buffer> counter_;  // one atomic uint, shared storage
    frame::Extent size_;
    std::uint64_t since_ = 0;  // accumulated_since of the frames it holds
    std::uint64_t next_ = 0;   // the index of the frame it expects next
};

}  // namespace serenity::metal
