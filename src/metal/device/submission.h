#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
// is shared, and every handler holds its own reference (R.20, CP.32). A late
// handler writes into state it keeps alive, never into freed memory.
//
// Settling. A submission is settled once its event and its feedback have
// both arrived and it did not fail: by begin(), for the submission whose slot
// it reuses; by wait_until_complete(), for the one it waits on; or by
// finish(), for every one left. Each submission is settled exactly once, by
// whichever comes first, and that call hands back its Completed record: the
// sequence, and the instants the feedback reports for when the GPU began and
// finished the work, in host time (MTL4CommitFeedback.h: GPUStartTime and
// GPUEndTime, seconds on the CPU's time base). This object records them and
// interprets nothing; what they mean as a frame time is Measurement's
// (core/measurement/frame_times.h). Which submissions are frames is the
// caller's to know, by their sequences.
//
// Residency. Metal 4 runs only on resources a residency set has made
// resident. keep_resident() adds an allocation to the set this queue uses
// and returns a Resident, which takes it out again when it is destroyed;
// resources are added at start-up and when a resize remakes an image, never
// per frame. The set retains what it holds (shown on this machine: an
// allocation's retain count rises when it is added and falls when it is
// removed), and a frame binds buffers by GPU address, which nothing retains.
//
// Lifetime: nothing is freed while the GPU may still use it (C.31, GPU.9:
// memory is reused "only after ... the old resource has no future users").
// Every object that owns GPU memory a frame uses is built after this
// Submission and destroyed before it, so on an error path it dies while
// frames may still be in flight. So a Resident, when destroyed, first waits
// for every committed submission to complete (wait_idle(), bounded, never
// throwing), then takes its allocation out of the set, then lets it go; and
// the objects that own GPU objects outside the set (the renderer's pipelines
// and argument table, the window's layer) wait the same way in their
// destructors before anything of theirs is released. After the normal end of
// a run (finish()) the wait returns at once. A submission still open when an
// allocation leaves the set may have recorded a use of it, so it can no
// longer be committed: commit() and present() then throw Error.
//
// Cost of a frame, from the CPU: one event wait (it returns at once unless
// the GPU is two frames behind), one allocator reset, one command buffer
// begun and ended, one commit, one event signal. Nothing of ours is allocated
// per frame (MEM.9). Metal 4 takes the feedback handler through commit
// options, an object made per commit, and metal-cpp hands the handler on as
// a block that Metal copies to the heap with what it captures: a few small
// allocations a frame, by the API's shape. Kept: an options object cannot
// shed a handler once added, so it cannot be reused for the next commit
// with a new one, and a few allocations against a frame of milliseconds are
// not worth a design of their own (Per.1, Per.6: no measured cost).
inline constexpr std::uint32_t frames_in_flight = 2;

// A submission settled (see Settling, above).
struct Completed {
    std::uint64_t sequence = 0;
    frame::Seconds gpu_start{0.0};  // host time
    frame::Seconds gpu_end{0.0};    // host time
};

// A submission begun: the command buffer to record into, its slot, and its
// sequence number, for wait_until_complete(). `settled` is the earlier
// submission that begin() settled to free the slot, if it was not settled
// already.
struct FrameSlot {
    MTL4::CommandBuffer* commands = nullptr;
    std::uint32_t slot = 0;
    std::uint64_t sequence = 0;
    std::optional<Completed> settled;
};

class Submission;

// An allocation resident through a Submission for as long as this object
// lives (R.1, C.31; see Residency and Lifetime, above). Move-only. Holds its
// own reference to the allocation, so the memory outlives the wait however
// the owner orders its members. The Submission must outlive it.
class Resident {
public:
    Resident() = default;
    Resident(const Resident&) = delete;
    Resident& operator=(const Resident&) = delete;
    Resident(Resident&& other) noexcept;
    Resident& operator=(Resident&& other) noexcept;
    ~Resident();

    // Takes the allocation out of the set now, after waiting for the GPU,
    // as destruction does; nothing is held after.
    void reset() noexcept;

private:
    friend class Submission;
    Resident(Submission& submission, MTL::Allocation* allocation);

    Submission* submission_ = nullptr;
    NS::SharedPtr<MTL::Allocation> allocation_;
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
    // throws Error if it failed. For readback, and for start-up work that
    // must finish before frames begin. Returns its record if this call
    // settled it, none if an earlier call had.
    std::optional<Completed> wait_until_complete(std::uint64_t sequence);

    // Blocks until every committed submission has completed and reported,
    // and throws Error if any failed. Called once, at the end of a run, so a
    // failure in its last frames is not lost. Returns the records of those it
    // settled, in sequence order. The destructor waits too, but a destructor
    // cannot report.
    std::vector<Completed> finish();

    // As finish(), but submissions may begin again after it: for replacing a
    // resource that frames in flight may still read (an accumulated image
    // remade at a new size, metal/frame/accumulation.h). A full wait,
    // visible in the call; never per frame.
    std::vector<Completed> drain();

    // Whether submission `sequence` has completed on the GPU, without
    // waiting: its event has been signalled, so what it wrote to shared
    // memory may be read. Says nothing of failure, which settling reports.
    bool has_completed(std::uint64_t sequence) const noexcept;

    // The sequence the next begin() hands out: every submission from here on
    // has this sequence or a later one.
    std::uint64_t next_sequence() const noexcept { return next_; }

    // Blocks until every committed submission has completed, or the timeout
    // passes; returns whether they all completed. Never throws, and reports
    // no failure (settling does): for destructors, which must not free what
    // the GPU still uses (see Lifetime, above).
    bool wait_idle() noexcept;

    // Makes `allocation` resident for every frame from now on, until the
    // returned Resident is destroyed. Throws Error if it is null.
    [[nodiscard]] Resident keep_resident(MTL::Allocation* allocation);

    // Makes `allocation` resident for as long as this Submission lives: for
    // memory that lives as long (a test's probe buffers). Throws Error if it
    // is null.
    void make_resident(MTL::Allocation* allocation);

    // Adds a residency set another object owns (a layer's) to the queue, and
    // takes it out again: remove_residency_set() first waits as wait_idle()
    // does, for the owner's destructor. Throws Error if `set` is null.
    void add_residency_set(MTL::ResidencySet* set);
    void remove_residency_set(MTL::ResidencySet* set) noexcept;

private:
    friend class Resident;

    // Waits as wait_idle() does, then takes `allocation` out of the set; see
    // Lifetime, above.
    void retire(MTL::Allocation* allocation) noexcept;

    // What a slot's feedback handlers write, on Metal's feedback queue, and
    // this object reads. `arrived` is the sequence + 1 of the last submission
    // whose feedback has arrived. `failure` is written before `failed` is
    // released, and only once. One handler runs per slot at a time: a slot is
    // not reused until its last handler has stored `arrived`.
    struct Feedback {
        std::atomic<std::uint64_t> arrived{0};
        std::atomic<bool> failed{false};
        std::string failure;
        frame::Seconds gpu_start{0.0};  // host time; written before `arrived` is released
        frame::Seconds gpu_end{0.0};
    };

    struct Slot {
        NS::SharedPtr<MTL4::CommandAllocator> allocator;
        NS::SharedPtr<MTL4::CommandBuffer> commands;
        std::shared_ptr<Feedback> feedback = std::make_shared<Feedback>();
        // The sequence + 1 of the last submission in this slot settled; read
        // and written only on the caller's thread.
        std::uint64_t settled_through = 0;
    };

    // Waits for submission `sequence`'s event, then its feedback, and throws
    // if either does not come in time or the submission failed. Returns its
    // record if this call settled it, none if it was settled already.
    std::optional<Completed> settle(std::uint64_t sequence);
    void end_and_commit(const MTL::Drawable* drawable);

    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL4::CommandQueue> queue_;
    NS::SharedPtr<MTL::ResidencySet> residency_;
    NS::SharedPtr<MTL::SharedEvent> completed_;
    std::array<Slot, frames_in_flight> slots_;
    std::uint64_t next_ = 0;   // the sequence begin() hands out next
    bool open_ = false;        // a submission is begun and not yet committed
    bool finished_ = false;    // finish() has run; nothing may be begun after
    bool abandoned_ = false;   // an allocation left the set while open_: never commit it
};

}  // namespace serenity::metal
