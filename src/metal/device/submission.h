#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "core/frame/frame_inputs.h"
#include "metal/device/device.h"

namespace serenity::metal {

// Axis: GPU backend.
//
// How a frame's work reaches the GPU, through Metal 4: one command queue,
// and the per-frame resources that let the CPU record frame N + 1 while the
// GPU runs frame N (GPU.7).
//
// The frames in flight. Two, so recording and execution overlap without the
// latency and memory a deeper ring adds (GPU.7's caveat). Each slot owns a
// command allocator and a command buffer, made once at construction (MEM.9,
// GPU.9). Submissions are counted here, from 0, apart from the frame's own
// index (frame/frame_inputs.h): the headless renderer may start at any frame,
// but the submissions it makes still count 0, 1, 2. Submission n uses slot
// n % frames_in_flight, and one shared event counts completed submissions:
// submission n signals n + 1 when the GPU has finished it.
//
// The protocol, in order, every frame:
//   begin()       waits until submission n - frames_in_flight has completed,
//                 so its slot is free; resets the slot's allocator; begins its
//                 command buffer; returns the buffer, the slot and n.
//   commit()      ends the command buffer, commits it, signals the event.
//   or present(d) the same, for a frame that writes drawable d: the queue
//                 waits for d before the work and signals it after, then d is
//                 presented. Pacing comes from the drawable: acquiring one
//                 blocks until the display frees it (presenter.h).
// Each begin() is followed by exactly one commit() or present() before the
// next begin(). Anything else is a programming error, checked: it throws
// Error.
//
// Anything else the CPU writes for a frame and the GPU reads (frame constants,
// renderer.h) is ringed by the same slot, so it is never overwritten while the
// GPU may still read it.
//
// The CPU never waits for the whole queue, only for the one submission whose
// slot it needs (GPU.7). The one deliberate full wait is
// wait_until_complete(n), for the headless renderer's readback (GPU.1):
// visible in the call, and never on the window's path.
//
// Failure is visible (E.2, E.14). Metal 4 reports a submission's GPU error
// through commit feedback, on a queue of its own, separately from the event
// that says the GPU is done. So a slot is reused only once both have arrived:
// begin() waits for the event and then for the feedback of the submission
// that last used the slot, and throws Error naming it if it failed;
// wait_until_complete() does the same for the submission it waits on, and
// finish() for every submission still unchecked, so a failure in the last
// frames before shutdown is reported too. A submission whose event or
// feedback does not arrive within the timeout throws Error as well, rather
// than blocking forever on a GPU that has stopped.
//
// The feedback handlers run on Metal's queue, possibly after this object is
// gone, so what they write is not in this object: each slot's feedback state
// is shared, and every handler holds its own reference (R.20, CP.3). A late
// handler writes into state it keeps alive, never into freed memory.
//
// Timing. The feedback also carries the GPU's start and end of the
// submission's work, on the GPU's own clock: their difference is the frame's
// GPU time (GPU.10), kept per slot and read with gpu_time() once the
// submission has settled. It is a duration on one clock and is never
// compared with the CPU's (TLM.11).
//
// Residency. Metal 4 runs only on resources a residency set has made
// resident. make_resident() adds an allocation to the set this queue uses;
// resources are added once, at start-up, never per frame.
//
// Cost of a frame, from the CPU: one event wait (it returns at once unless
// the GPU is two frames behind), one allocator reset, one command buffer
// begun and ended, one commit, one event signal. Nothing of ours is allocated
// per frame (MEM.9). Metal 4 takes the feedback handler through commit
// options, an object made per commit: one small allocation a frame that is
// the API's shape, not ours, and is kept.
inline constexpr std::uint32_t frames_in_flight = 2;

// A submission begun: the command buffer to record into, its slot, and its
// sequence number, for wait_until_complete().
struct FrameSlot {
    MTL4::CommandBuffer* commands = nullptr;
    std::uint32_t slot = 0;
    std::uint64_t sequence = 0;
};

class Submission {
public:
    explicit Submission(const Device& device);

    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;
    Submission(Submission&&) = delete;
    Submission& operator=(Submission&&) = delete;
    ~Submission();

    // Begins the next submission. Its command buffer is valid until the
    // matching commit() or present().
    FrameSlot begin();

    // Ends and commits the frame begun last.
    void commit();

    // Ends and commits the frame begun last, which writes `drawable`'s
    // texture, and presents it.
    void present(CA::MetalDrawable* drawable);

    // Blocks until submission `sequence` has completed on the GPU, and
    // throws Error if it failed. For readback.
    void wait_until_complete(std::uint64_t sequence);

    // Blocks until every committed submission has completed and reported,
    // and throws Error if any failed. Called once, at the end of a run, so a
    // failure in its last frames is not lost. The destructor waits too, but
    // a destructor cannot report.
    void finish();

    // The GPU time of the submission settled most recently: by begin(), which
    // settles the one whose slot it reuses, by wait_until_complete() or by
    // finish(). None until one has settled.
    std::optional<frame::Seconds> gpu_time() const { return last_gpu_time_; }

    // Makes `allocation` resident for every frame from now on.
    void make_resident(MTL::Allocation* allocation);

    // The queue, for adding a residency set another object owns (a layer's).
    MTL4::CommandQueue* queue() const { return queue_.get(); }

private:
    // What a slot's feedback handlers write, on Metal's feedback queue, and
    // this object reads. `arrived` is the sequence + 1 of the last submission
    // whose feedback has arrived. `failure` is written before `failed` is
    // released, and only once. One handler runs per slot at a time: a slot is
    // not reused until its last handler has stored `arrived`.
    struct Feedback {
        std::atomic<std::uint64_t> arrived{0};
        std::atomic<bool> failed{false};
        std::string failure;
        double gpu_seconds = 0.0;  // written before `arrived` is released
    };

    struct Slot {
        NS::SharedPtr<MTL4::CommandAllocator> allocator;
        NS::SharedPtr<MTL4::CommandBuffer> commands;
        std::shared_ptr<Feedback> feedback = std::make_shared<Feedback>();
    };

    // Waits for submission `sequence`'s event, then its feedback, and throws
    // if either does not come in time or the submission failed.
    void settle(std::uint64_t sequence);
    void end_and_commit(const MTL::Drawable* drawable);

    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL4::CommandQueue> queue_;
    NS::SharedPtr<MTL::ResidencySet> residency_;
    NS::SharedPtr<MTL::SharedEvent> completed_;
    std::array<Slot, frames_in_flight> slots_;
    std::uint64_t next_ = 0;   // the sequence begin() hands out next
    bool open_ = false;        // a submission is begun and not yet committed
    bool finished_ = false;    // finish() has run; nothing may be begun after
    std::optional<frame::Seconds> last_gpu_time_;
};

}  // namespace serenity::metal
