#pragma once

#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Frame graph (history across frames: contract 6, in its first form),
// on the GPU.
//
// The image a converging pass averages its frames into: for each pixel, the
// mean radiance of its samples from frames accumulated_since .. index - 1
// (frame/frame_inputs.h) and their count (metal/film/accumulate.metal.h
// says what a pixel holds), in RGBA32Float, at the frame's size. The pass
// reads it, folds the new frame in, and writes it back (metal/passes/path/
// path.h, by the rule in metal/film/accumulate.metal.h).
//
// Which frames it holds, and whether a frame may join it, is the core's rule
// (core/frame/history.h), kept by a frame::History this object owns: the
// backend decides nothing about it (principle 10). What this object does is
// the GPU's side: make the image, at the size the rule says, and keep it
// resident.
//
// prepare() is called once a frame, before the frame's submission begins
// (metal/frame/renderer.h, Renderer::prepare). It asks the rule whether the
// frame joins, and returns the count of frames the image holds before it;
// where the rule says the image is made anew (a first frame, or a start over
// at another size) it makes it. A frame the rule refuses is refused here, by
// Error with the rule's reason, and nothing changes; so is a size Metal
// cannot make an image of (metal/device/device.h, max_texture_side). The
// rule is asked on a copy of the history, kept once the image exists, so
// the history never claims an image that is not there (E.4).
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
    // of a scene that changes or not; see above. Throws Error if the core's
    // rule refuses the frame, or the image cannot be made.
    std::uint32_t prepare(const frame::FrameInputs& inputs, frame::Extent size, bool scene_changes);

    // The image, valid until the next prepare(); null before the first.
    MTL::Texture* texture() const noexcept { return texture_.get(); }

private:
    NS::SharedPtr<MTL::Device> device_;
    Submission& submission_;
    NS::SharedPtr<MTL::Texture> texture_;
    Resident resident_;  // released after the GPU is done with it (submission.h)
    frame::History history_;  // the core's rule, and what the image holds
};

}  // namespace serenity::metal
