#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/device/device.h"
#include "metal/device/resident.h"
#include "metal/device/submission.h"

namespace serenity::metal {

// Axis: Film (what a pixel accumulates: its failures).
//
// A count of one kind of failure the frame's kernels report on the GPU, so
// that a failure is counted where it happens and shown, never lost
// silently. The renderer keeps two (metal/frame/renderer.h):
//
//   - samples a converging pass left out of its pixels for not being
//     finite (metal/film/accumulate.metal.h), non_finite_samples();
//   - MNEE requests the work list had no room for, whose light is lost
//     (contract 16, core/contracts/mnee_request.h; the MNEE pass's grid
//     kernel counts them), mnee_lost_requests().
//
// Film decides what a failed sample is and keeps the counts; the frame graph
// only binds them, and the window and the headless renderer report them.
// Each is counted apart: a lost request is not a sample that is not finite,
// and was once reported as one, through a NaN that fast math need not keep
// (docs/research/2026-10-16-codebase-guidelines-pass.md, C14).
//
// One 32-bit counter per frame in flight, in shared memory, indexed by the
// submission's slot (submission.h), which a frame's kernels add to on the
// GPU. The CPU reads a slot's counter only once the frame that wrote it has
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
// A frame's counter holds at most its pixels' samples, 7.7 million at the
// display's size, or its requests, at most one a surface a path reaches,
// 256 at the path tracer's safety stop (metal/integrator/path.metal.h):
// 2.0 x 10^9 there, under 2^32. The total is 64-bit, so neither overflows.
//
// Throws MetalError if the device cannot make the counters. Not
// performance-sensitive: a few bytes, once a frame.
class FailureCount {
public:
    // `what` names what it counts, for its errors: "samples not finite".
    FailureCount(const Device& device, Submission& submission, std::string_view what);

    FailureCount(const FailureCount&) = delete;
    FailureCount& operator=(const FailureCount&) = delete;
    FailureCount(FailureCount&&) = delete;
    FailureCount& operator=(FailureCount&&) = delete;
    ~FailureCount() = default;

    // Before frame `sequence` is recorded into `slot`; see above.
    void begin_frame(std::uint32_t slot, std::uint64_t sequence);

    // Slot `slot`'s counter, as a shader binds it. Throws MetalError for a slot
    // that is not one, as begin_frame() does.
    MTL::GPUAddress address(std::uint32_t slot) const;

    // The failures over every completed frame; see above.
    std::uint64_t count() const noexcept;

private:
    std::uint32_t* counter(std::uint32_t slot) noexcept;
    const std::uint32_t* counter(std::uint32_t slot) const noexcept;

    const Submission& submission_;
    std::string what_;
    ResidentBuffer counters_;  // one uint per slot
    std::array<std::uint64_t, frames_in_flight> counting_{};  // sequence + 1 each slot counts for; 0 for none
    std::uint64_t total_ = 0;
};

}  // namespace serenity::metal
