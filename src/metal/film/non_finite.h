#pragma once

#include <array>
#include <cstdint>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Film (what a pixel accumulates: its failures).
//
// The count of samples a converging pass left out of its pixels for not
// being finite (metal/film/accumulate.metal.h). Film decides what a failed
// sample is and keeps the count; the frame graph only binds it
// (metal/frame/renderer.h), and the window and the headless renderer report
// it.
//
// One 32-bit counter per frame in flight, in shared memory, indexed by the
// submission's slot (submission.h), which a frame's pass adds to on the GPU.
// The CPU reads a slot's counter only once the frame that wrote it has
// completed: Metal makes the GPU's writes to shared memory visible to the
// CPU only then. So:
//
//   - begin_frame(slot, sequence), as a frame is recorded into a slot,
//     first collects the slot's counter into a 64-bit total kept on the CPU
//     (the slot's last frame has completed: Submission::begin() waited for
//     it), zeroes it for the new frame, and remembers which frame the slot
//     now counts for;
//   - count() is the total, plus each slot's counter whose frame has
//     completed (Submission::has_completed, which does not wait), so it is
//     exact for every completed frame and never reads a counter a frame in
//     flight may be writing.
//
// A frame's counter holds at most its pixels, 7.7 million at the display's
// size, and the total is 64-bit, so neither overflows.
//
// Throws Error if the device cannot make the counters. Not
// performance-sensitive: a few bytes, once a frame.
class NonFinite {
public:
    NonFinite(const Device& device, Submission& submission);

    NonFinite(const NonFinite&) = delete;
    NonFinite& operator=(const NonFinite&) = delete;
    NonFinite(NonFinite&&) = delete;
    NonFinite& operator=(NonFinite&&) = delete;
    ~NonFinite() = default;

    // Before frame `sequence` is recorded into `slot`; see above.
    void begin_frame(std::uint32_t slot, std::uint64_t sequence);

    // Slot `slot`'s counter, as a shader binds it. Throws Error for a slot
    // that is not one, as begin_frame() does.
    MTL::GPUAddress address(std::uint32_t slot) const;

    // The samples left out over every completed frame; see above.
    std::uint64_t count() const;

private:
    std::uint32_t* counter(std::uint32_t slot) const;

    const Submission& submission_;
    NS::SharedPtr<MTL::Buffer> counters_;  // one uint per slot
    Resident resident_;
    std::array<std::uint64_t, frames_in_flight> counting_{};  // sequence + 1 each slot counts for; 0 for none
    std::uint64_t total_ = 0;
};

}  // namespace serenity::metal
