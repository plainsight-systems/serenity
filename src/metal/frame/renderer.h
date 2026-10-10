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
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_resources.h"
#include "metal/film/non_finite.h"
#include "metal/frame/accumulation.h"
#include "metal/frame/frame_images.h"
#include "metal/passes/display/display.h"
#include "metal/passes/path/path.h"
#include "metal/passes/preview/preview.h"
#include "metal/passes/test_pattern/test_pattern.h"
#include "metal/passes/tone_map/tone_map.h"
#include "metal/scene/shape_transforms.h"
#include "metal/scene/light_glows.h"
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
// its acceleration structures (scene_acceleration.h) at construction, and refuses, by
// Error, a graph that needs a scene when given none. Each frame then gives
// every pass the same resources (frame_resources.h), the scene among them.
// A graph that reads no scene runs with or without one; given one, it is not
// copied, and the structure is built over the scene's shapes as the Shape
// family bounds them (shapes/shapes.h).
//
// Animate (logical-overview.md). The shapes' transforms and the lights'
// glows are kept apart from the scene's still arrays
// (metal/scene/shape_transforms.h, metal/scene/light_glows.h), and the
// acceleration structure holds each shape as its geometry's bounds placed
// by its transform (scene_acceleration.h). When the scene changes
// (core/animation/animate.h), recording a frame begins, before any pass, by
// placing every moving shape and lighting every glowing light at the
// frame's time, by the core (animation::animate), into the frame slot's
// transforms and glows; and, when shapes move, by recording the slot's
// structure's build, from the moving shapes' new boxes, and its barrier into
// the frame's encoder (SceneAcceleration::update). Every pass of the frame
// then sees the scene at that time: the slot's transforms, glows and
// structure. The CPU's work is the moving shapes' and the glowing lights'
// alone; the rest is not rewritten. Animate runs whenever the scene changes
// (animation::changes), so a firefly that blinks in place blinks, and the
// history rule is asked whether the scene changes (core/frame/history.h);
// the structure is rebuilt only when shapes move (animation::moves). Every
// pass that reads the scene binds the frame's transforms and glows beside
// the scene's block (metal/passes/bindings.h), where the emitter reads them
// (metal/lights/emitter.metal.h). The scene's motions are
// the core's to evaluate; the renderer only gives them the memory to write
// into and the time (principle 10). It is also where the families meet: it
// tells the acceleration structure which shapes move, by the movers'
// targets, and nothing else of the scene's animation. A still scene does
// none of this.
//
// The camera. It is the frame's, an input like its time (frame_inputs.h):
// the caller chooses it, and the renderer holds none. Each frame the
// renderer frames the camera it is given for the target's size
// (camera/thin_lens.h) and binds it; a frame of a graph that reads a scene and
// has no camera is refused, by Error.
//
// The accumulated image. A frame graph with a pass that averages its frames
// (frame::accumulates) has one (accumulation.h), and at most one such pass,
// a rule of the core's (frame::invalid, core/frame/schedule.h) that the
// renderer applies at construction, refusing by Error any schedule the core
// would not accept. Each frame, prepare()
// readies it from the frame's inputs before the frame's submission begins,
// since remaking it for a new size waits for the frames in flight; record()
// then gives the pass the image and the count of frames it holds, through the
// frame constants, and Film's counter of samples left out for not being finite
// (metal/film/non_finite.h), which non_finite_samples() reads. A frame whose
// inputs claim an image the history does not hold is refused, by Error, and so
// is a frame recorded without being prepared, and, when the scene changes, a
// frame at another time than the frames the image holds (accumulation.h).
//
// The target is a texture and its size, whatever owns it: the window's
// drawable (presenter.h) or an offscreen image (offscreen.h). The renderer
// cannot tell which, so a frame is the same function of its inputs in the
// window and headless (principle 1).
//
// The images between passes (frame_images.h): the radiance image, which a
// light pass writes and a presenting pass reads, and the tone-map pass's
// bloom pyramid, made by prepare() at the frame's size when the schedule
// uses them (core/frame/schedule.h: writes_radiance, tone_map).
//
// Metal 4 does not track hazards between passes. When a pass reads what an
// earlier pass in the frame wrote, the renderer records the barrier between
// them, so the dependencies are explicit and in one place (GPU.7): before a
// pass that reads the radiance image, a barrier from dispatch to dispatch
// after the pass that wrote it. Within a pass of several dispatches, the
// pass records its own (passes/tone_map/tone_map.h). Between frames, the
// images between passes are one set the frames in flight share
// (frame_images.h), and the accumulated image is the frame before's history:
// before the first pass of a frame that writes the images or accumulates,
// the renderer records one barrier waiting for the queue's earlier
// dispatches (barrierAfterQueueStages, dispatch before dispatch), so a frame
// never writes the images while the frame before still reads them, nor reads
// the accumulated image before the frame before has written it. Each
// barrier names its hazard and none is recorded twice (GPU.8: barriers
// describe real hazards, and the frame graph, which knows every pass,
// synthesizes them). The queue's wait
// for the drawable orders the frame against the display
// (submission.h).
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
// (GPU.6). When shapes move, also: each moving shape's motion, its
// transform's write and its box's (core/animation/animate.h,
// scene_acceleration.h), and one structure build and one barrier.
// Nothing is allocated.
class Renderer {
public:
    // Builds a pass for each entry of `schedule` from the backend's shader
    // library, compiled into the program (cmake/MetalLibrary.cmake), makes the
    // ring resident through `submission`, and, if any pass reads the scene,
    // puts `scene` on the GPU. `scene` may be null when no pass reads it.
    // Throws Error if the schedule is invalid (frame::invalid), a pass needs a
    // scene and there is none, a glowing light is not one of the scene's, or
    // a pipeline, buffer or structure cannot be built.
    Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule,
             const scene::SceneDescription* scene);

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(Renderer&&) = delete;
    // Waits for the GPU before anything of the renderer's is released
    // (submission.h, Lifetime).
    ~Renderer();

    // Checks frame `inputs` and readies what it needs from before it: its
    // time must be finite within float's range, which the shaders read; a
    // graph that reads a scene needs the frame's camera; the accumulated
    // image, if the graph has one (accumulation.h), must hold what the
    // inputs claim; and the images between passes, if the graph has any
    // (frame_images.h), are made at `size`. Called before the frame's
    // submission begins, for every graph; throws Error if any fails.
    //
    // Everything a frame's inputs or the scene can get wrong is checked here
    // or at construction, before the submission begins (E.4): an exception
    // after begin() leaves the submission open, never to be committed
    // (submission.h), so record() throws only for a broken protocol or a
    // Metal failure, which end the run.
    void prepare(const frame::FrameInputs& inputs, frame::Extent size);

    // Records frame `inputs` into `frame`, writing `target`, of `size`.
    // `frame` is what Submission::begin() returned. Throws Error if the
    // frame was not prepared at this size, if the graph uses images or
    // accumulates; if it has no camera and the graph reads a scene; or if
    // Metal makes no encoder.
    void record(const FrameSlot& frame, const frame::FrameInputs& inputs, MTL::Texture* target,
                frame::Extent size);

    // Samples left out for not being finite (metal/film/non_finite.h), over
    // every completed frame; 0 for a graph that accumulates nothing. Reads
    // no counter a frame in flight may be writing.
    std::uint64_t non_finite_samples() const;

private:
    using Pass = std::variant<TestPatternPass, PreviewPass, PathPass, DisplayPass, ToneMapPass>;

    Submission& submission_;
    Library library_;
    bool needs_scene_ = false;  // some pass in the schedule reads the scene
    std::unique_ptr<SceneBuffers> scene_;
    std::unique_ptr<SceneAcceleration> acceleration_;
    std::unique_ptr<ShapeTransforms> transforms_;
    std::unique_ptr<LightGlows> glows_;
    animation::Animation animation_;  // empty unless the scene changes
    std::unique_ptr<FrameImages> images_;         // when a pass writes radiance
    std::unique_ptr<Accumulation> accumulation_;  // when a pass accumulates
    std::unique_ptr<NonFinite> non_finite_;       // with it
    struct Prepared {
        std::uint64_t index = 0;
        frame::Extent size;
        std::uint32_t accumulated_frames = 0;
    };
    std::optional<Prepared> prepared_;  // the frame prepare() readied, until recorded
    NS::SharedPtr<MTL::Buffer> constants_;
    Resident constants_resident_;
    std::array<NS::SharedPtr<MTL4::ArgumentTable>, frames_in_flight> arguments_;
    std::vector<frame::PassKind> kinds_;  // the schedule's
    std::vector<Pass> passes_;            // in schedule order, one per kind
    static constexpr std::size_t no_pass = static_cast<std::size_t>(-1);
    // The first pass that writes the images between passes or accumulates,
    // before which the queue barrier is recorded; no_pass if none does.
    std::size_t first_cross_frame_ = no_pass;
};

// One frame, start to finish, for each kind of target. The window and the
// headless renderer call these and nothing below them, so neither names a
// Metal type (file-mapping.md).
//
// Both prepare the frame (Renderer::prepare) before its submission begins.
//
// render_to_window() acquires a drawable, records the frame into it and
// presents it, inside an autorelease pool of its own (the drawable is an
// autoreleased object). It returns the frame's sequence and the earlier
// submission its begin() settled, if any (submission.h), for the window to
// measure; or none, having done nothing, when Core Animation had no drawable
// to give (presenter.h).
//
// render_to_window() refuses, by Error, a drawable whose texture is not the
// size the layer was given: the passes would write past it.
//
// render_to_offscreen() records the frame into `target` and commits it,
// without waiting; it returns the submission's sequence, for
// Submission::wait_until_complete() before reading `target` back. The
// headless renderer measures nothing, so what its begin() settled is not
// returned. It drains an autorelease pool of its own too, so a caller with
// none (the headless renderer) leaks nothing a frame.
struct WindowFrame {
    std::uint64_t sequence = 0;
    std::optional<Completed> settled;
};
std::optional<WindowFrame> render_to_window(Submission& submission, Presenter& presenter, Renderer& renderer,
                                            const frame::FrameInputs& inputs);
std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target, Renderer& renderer,
                                  const frame::FrameInputs& inputs);

}  // namespace serenity::metal
