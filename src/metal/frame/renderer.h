#pragma once

#include <array>
#include <variant>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/passes/test_pattern/test_pattern.h"

namespace serenity::metal {

// Axis: Frame graph.
//
// Carries out the core's schedule on Metal (core/frame/schedule.h): given a
// frame's inputs and the image it renders into, records the schedule's passes,
// in the schedule's order, into the frame's command buffer. It decides
// nothing about what a frame computes; that is the schedule's
// (logical-overview.md, principle 10).
//
// At construction each of the schedule's pass kinds becomes this backend's
// pass for it, with its pipeline built. The mapping is a switch with no
// default, so a kind this backend does not implement fails the build, not a
// frame. The passes are held by value, in schedule order, in a variant over
// this backend's pass types: no virtual dispatch and no allocation per frame.
//
// The target is a texture and its size, whatever owns it: the window's
// drawable (presenter.h) or an offscreen image (offscreen.h). The renderer
// cannot tell which, so a frame is the same function of its inputs in the
// window and headless (principle 1).
//
// Metal 4 does not track hazards between passes. When a pass reads what an
// earlier pass wrote, the renderer records the barrier between them, so the
// dependencies are explicit and in one place (GPU.7). Today every pass writes
// only the target and none reads another's output, so there is none; the
// queue's wait for the drawable orders the frame against the display
// (submission.h).
//
// Frame constants (contracts/frame_constants.h) reach the shaders through a
// buffer, because Metal 4's compute encoder has no inline constants. The
// buffer is a ring, one slot per frame in flight, indexed by the
// submission's slot (submission.h), so the CPU writes the next frame's
// constants while the GPU may still read the last one's (GPU.7); slots are
// 256 bytes apart. The argument tables the passes bind through are ringed
// the same way, one per slot: Apple's documentation does not say whether an
// encoder copies a table's bindings at each dispatch, so no frame in flight
// shares a table with the next. Within a frame, passes rebind the slot's
// table between dispatches, which tests/gpu/argument_table_test.cpp shows
// Metal 4 honours on this machine. The ring and the tables are made once, at
// construction (MEM.9).
//
// Cost of recording a frame, on the CPU: 16 bytes written to the ring, one
// compute encoder, and per pass one pipeline bind, its argument-table
// entries and one dispatch, all in one command buffer (GPU.6). Nothing is
// allocated.
class Renderer {
public:
    // Builds a pass for each entry of `schedule` from the backend's shader
    // library, compiled into the program (cmake/MetalLibrary.cmake), and
    // makes the ring resident through `submission`. Throws Error if the
    // schedule is empty or a pipeline cannot be built.
    Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule);

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;
    ~Renderer() = default;

    // Records frame `inputs` into `frame`, writing `target`, of `size`.
    // `frame` is what Submission::begin() returned.
    void record(const FrameSlot& frame, const frame::FrameInputs& inputs, MTL::Texture* target,
                frame::Extent size);

private:
    using Pass = std::variant<TestPatternPass>;

    Library library_;
    NS::SharedPtr<MTL::Buffer> constants_;
    std::array<NS::SharedPtr<MTL4::ArgumentTable>, frames_in_flight> arguments_;
    std::vector<Pass> passes_;  // in schedule order
};

// One frame, start to finish, for each kind of target. The window and the
// headless renderer call these and nothing below them, so neither names a
// Metal type (file-mapping.md).
//
// render_to_window() acquires a drawable, records the frame into it and
// presents it, inside an autorelease pool of its own (the drawable is an
// autoreleased object). It returns false, having done nothing, when Core
// Animation had no drawable to give (presenter.h).
//
// render_to_offscreen() records the frame into `target` and commits it,
// without waiting; it returns the submission's sequence, for
// Submission::wait_until_complete() before reading `target` back.
bool render_to_window(Submission& submission, Presenter& presenter, Renderer& renderer,
                      const frame::FrameInputs& inputs);
std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target, Renderer& renderer,
                                  const frame::FrameInputs& inputs);

}  // namespace serenity::metal
