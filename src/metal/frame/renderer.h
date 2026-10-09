#pragma once

#include <array>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "metal/acceleration/primitives.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_resources.h"
#include "metal/frame/accumulation.h"
#include "metal/passes/path/path.h"
#include "metal/passes/preview/preview.h"
#include "metal/passes/test_pattern/test_pattern.h"
#include "metal/scene/scene_buffers.h"

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
// The scene. A frame graph whose passes read a scene (frame::needs_scene)
// needs one: the renderer copies it to the GPU (scene_buffers.h) and builds
// its acceleration structure (primitives.h) at construction, and refuses, by
// Error, a graph that needs a scene when given none. Each frame then gives
// every pass the same resources (frame_resources.h), the scene among them.
// A graph that reads no scene runs with or without one; given one, it is not
// copied, and the structure is built over the scene's shapes as the Shape
// family bounds them (shapes/shapes.h).
//
// The camera. It is the frame's, an input like its time (frame_inputs.h):
// the caller chooses it, and the renderer holds none. Each frame the
// renderer frames the camera it is given for the target's size
// (camera/pinhole.h) and binds it; a frame of a graph that reads a scene and
// has no camera is refused, by Error.
//
// The accumulated image. A frame graph with a pass that averages its frames
// (frame::accumulates) has one (accumulation.h), which the renderer prepares
// each frame from the frame's inputs before recording it: the count of
// frames it holds goes to the frame constants, and a frame whose inputs
// claim an image the history does not hold is refused, by Error.
//
// The target is a texture and its size, whatever owns it: the window's
// drawable (presenter.h) or an offscreen image (offscreen.h). The renderer
// cannot tell which, so a frame is the same function of its inputs in the
// window and headless (principle 1).
//
// Metal 4 does not track hazards between passes. When a pass reads what an
// earlier pass in the frame wrote, the renderer records the barrier between
// them, so the dependencies are explicit and in one place (GPU.7). Today no
// pass reads another's output in the frame, so there is none. Between
// frames, a pass that reads its own history waits for the previous frame's
// dispatches itself (passes/path/path.h). The queue's wait for the drawable
// orders the frame against the display (submission.h).
//
// Frame constants (contracts/frame_constants.h) and the framed camera
// (contracts/camera.h) reach the shaders through a buffer, because Metal 4's
// compute encoder has no inline constants. The buffer is a ring, one slot per
// frame in flight, indexed by the submission's slot (submission.h), so the CPU
// writes the next frame's values while the GPU may still read the last one's
// (GPU.7). Slots are 256 bytes apart; in each, the frame constants are at 0
// and the camera at 128, so each contract is bound at its own address and
// either can grow without moving the other. The argument tables the passes
// bind through are ringed the same way, one per slot: Apple's documentation
// does not say whether an encoder copies a table's bindings at each dispatch,
// so no frame in flight shares a table with the next. Within a frame, passes
// rebind the slot's table between dispatches, which
// tests/gpu/argument_table_test.cpp shows Metal 4 honours on this machine. The
// ring and the tables are made once, at construction (MEM.9).
//
// Cost of recording a frame, on the CPU: 96 bytes written to the ring (32 of
// frame constants and, with a camera, 64 of camera), one framing of the camera
// (a few dozen flops), one compute encoder, and per pass one pipeline bind,
// its argument-table entries and one dispatch, all in one command buffer
// (GPU.6). Nothing is allocated.
class Renderer {
public:
    // Builds a pass for each entry of `schedule` from the backend's shader
    // library, compiled into the program (cmake/MetalLibrary.cmake), makes
    // the ring resident through `submission`, and, if any pass reads the
    // scene, puts `scene` on the GPU. `scene` may be null when no pass reads
    // it. Throws Error if the schedule is empty, a pass needs a scene and
    // there is none, or a pipeline, buffer or structure cannot be built.
    Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule,
             const scene::SceneDescription* scene);

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;
    ~Renderer() = default;

    // Records frame `inputs` into `frame`, writing `target`, of `size`.
    // `frame` is what Submission::begin() returned. Throws Error if the
    // schedule reads a scene and `inputs` has no camera.
    void record(const FrameSlot& frame, const frame::FrameInputs& inputs, MTL::Texture* target,
                frame::Extent size);

private:
    using Pass = std::variant<TestPatternPass, PreviewPass, PathPass>;

    Library library_;
    bool needs_scene_ = false;  // some pass in the schedule reads the scene
    std::unique_ptr<SceneBuffers> scene_;
    std::unique_ptr<PrimitiveAcceleration> acceleration_;
    std::unique_ptr<Accumulation> accumulation_;  // when a pass accumulates
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
// autoreleased object). It returns the frame's sequence and the earlier
// submission its begin() settled, if any (submission.h), for the window to
// measure; or none, having done nothing, when Core Animation had no drawable
// to give (presenter.h).
//
// render_to_offscreen() records the frame into `target` and commits it,
// without waiting; it returns the submission's sequence, for
// Submission::wait_until_complete() before reading `target` back. The
// headless renderer measures nothing, so what its begin() settled is not
// returned.
struct WindowFrame {
    std::uint64_t sequence = 0;
    std::optional<Completed> settled;
};
std::optional<WindowFrame> render_to_window(Submission& submission, Presenter& presenter, Renderer& renderer,
                                            const frame::FrameInputs& inputs);
std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target, Renderer& renderer,
                                  const frame::FrameInputs& inputs);

}  // namespace serenity::metal
