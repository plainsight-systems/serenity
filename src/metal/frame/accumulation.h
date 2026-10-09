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
// The image a converging pass averages its frames into: for each pixel, the
// mean radiance of its samples from frames accumulated_since .. index - 1
// (frame/frame_inputs.h) and their count (metal/film/accumulate.metal.h
// says what a pixel holds), in RGBA32Float, at the frame's size. The pass reads it, folds the new frame
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
//     count past 2^24 - 1, beyond which a pixel's count, a float, is no
//     longer exact (77 hours of frames at 60 a second): the caller starts
//     over before it (frame/frame_inputs.h). And so, when the scene moves
//     (core/animation/animate.h), is a frame at another time than the frames
//     the image holds: they are samples of one instant, and a frame of
//     another would average two scenes into a blur no camera sees. The
//     times must be equal exactly, as a frozen --time gives them
//     (headless/options.h); the window, whose time always advances, starts
//     over every frame while anything moves. A still scene looks the same at
//     every t, so its frames join whatever their times.
//
// The image is the GPU's alone, in private storage, made resident through
// the submission. Remaking it for a new size first drains the submission,
// which is why prepare() comes before a submission begins: frames in flight
// may still read the old image, so it leaves the residency set and is
// released only once they have completed. A resize is rare and the wait is a
// frame or two, so it is not on any budget.

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

    // The count of frames the image holds before frame `inputs`, at `size`,
    // of a scene that moves or not; see above. Throws Error if the image
    // cannot hold what the inputs claim, or cannot be made.
    std::uint32_t prepare(const frame::FrameInputs& inputs, frame::Extent size, bool scene_moves);

    // The image, valid until the next prepare(); null before the first.
    MTL::Texture* texture() const { return texture_.get(); }


private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    NS::SharedPtr<MTL::Texture> texture_;
    frame::Extent size_;
    std::uint64_t since_ = 0;  // accumulated_since of the frames it holds
    std::uint64_t next_ = 0;   // the index of the frame it expects next
    frame::Seconds time_{0.0}; // the time of the frames it holds, when the scene moves
};

}  // namespace serenity::metal
