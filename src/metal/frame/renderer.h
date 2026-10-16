#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
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
#include "metal/device/frame_array.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"
#include "metal/frame/frame_resources.h"
#include "metal/film/non_finite.h"
#include "metal/frame/accumulation.h"
#include "metal/frame/denoise_images.h"
#include "metal/frame/feature_images.h"
#include "metal/frame/frame_images.h"
#include "metal/frame/mnee_list.h"
#include "metal/passes/denoise/denoise.h"
#include "metal/passes/display/display.h"
#include "metal/passes/features/features.h"
#include "metal/passes/path/path.h"
#include "metal/passes/preview/preview.h"
#include "metal/passes/restir_initial/restir_initial.h"
#include "metal/passes/restir_pt_duplication/restir_pt_duplication.h"
#include "metal/passes/restir_pt_initial/restir_pt_initial.h"
#include "metal/passes/restir_pt_shade/restir_pt_shade.h"
#include "metal/passes/restir_pt_spatial/restir_pt_spatial.h"
#include "metal/passes/restir_pt_temporal/restir_pt_temporal.h"
#include "metal/passes/restir_shade/restir_shade.h"
#include "metal/passes/restir_spatial/restir_spatial.h"
#include "metal/passes/restir_temporal/restir_temporal.h"
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
// its acceleration structures (scene_acceleration.h) at construction, and
// refuses, by MetalError, a graph that needs a scene when given none. Each frame
// then gives
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
// (metal/lights/emitter.metal.h). The scene's motions are the core's to
// evaluate; the renderer only gives them the memory to write into and the
// time (principle 10). It is also where the families meet: it tells the
// acceleration structure which shapes move, by the movers' targets, and
// nothing else of the scene's animation. A still scene does none of this.
//
// The camera. It is the frame's, an input like its time (frame_inputs.h):
// the caller chooses it, and the renderer holds none. Each frame the
// renderer frames the camera it is given for the target's size
// (camera/thin_lens.h) and binds it; a frame of a graph that reads a scene
// and has no camera is refused, by MetalError, in prepare().
//
// The accumulated image. A frame graph with a pass that averages its frames
// (frame::accumulates) has one (accumulation.h), and at most one such pass,
// a rule of the core's (frame::invalid, core/frame/schedule.h) that the
// renderer applies at construction, refusing by MetalError any schedule the core
// would not accept. Each frame, prepare() readies it from the frame's inputs
// before the frame's submission begins, since remaking it for a new size
// waits for the frames in flight; record() then gives the pass the image and
// the count of frames it holds, through the frame constants, and Film's
// counter of samples left out for not being finite (metal/film/
// non_finite.h), which non_finite_samples() reads. A frame whose inputs
// claim an image the history does not hold is refused, by MetalError, and so is a
// frame recorded without being prepared, and, when the scene changes, a
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
// uses them (core/frame/schedule.h: writes_radiance, tone_map). When the
// light pass accumulates, the accumulated image is the radiance image, and
// the passes after it are given that (passes/path/path.h).
//
// Metal 4 does not track hazards between passes. When a pass reads what an
// earlier pass in the frame wrote, the renderer records the barrier between
// them, so the dependencies are explicit and in one place (GPU.7): before a
// pass that reads the radiance image, a barrier from dispatch to dispatch
// after the pass that wrote it. Within a pass of several dispatches, the
// pass records its own (passes/tone_map/tone_map.h). Between frames, the
// images between passes are one set the frames in flight share
// (frame_images.h), and the accumulated image is the frame before's history:
// before the first pass of a frame that writes the images, accumulates or
// is one of ReSTIR DI's (whose arrays, ReuseImages, the frames share too),
// the renderer records one barrier waiting for the queue's earlier
// dispatches (barrierAfterQueueStages, dispatch before dispatch), so a frame
// never writes the images while the frame before still reads them, nor reads
// the accumulated image before the frame before has written it. Each
// barrier names its hazard and none is recorded twice (GPU.8: barriers
// describe real hazards, and the frame graph, which knows every pass,
// synthesizes them). Without it the frames would overlap, which is slower,
// not faster (docs/research/2026-10-09-pass-costs.md). The queue's wait for
// the drawable orders the frame against the display (submission.h).
//
// Frame constants (contracts/frame_constants.h) and the framed camera
// (contracts/camera.h) reach the shaders through a buffer, because Metal 4's
// compute encoder has no inline constants. The buffer is a ring, one slot per
// frame in flight, indexed by the submission's slot (submission.h), so the CPU
// writes the next frame's values while the GPU may still read the last one's
// (GPU.7): a FrameArray of a copy per frame in flight (metal/device/
// frame_array.h), whose rule says when a slot may be written. Slots are 256
// bytes; in each, the frame constants are at 0 and the camera at 128, so
// each contract is bound at its own address and either can grow without
// moving the other. The passes bind through one
// argument table, which every frame shares: Metal "takes a snapshot of the
// resources in the argument table when you make dispatch or execute calls"
// (MTL4ComputeCommandEncoder.h, setArgumentTable), so rebinding it, between
// one pass's dispatches or for the next frame while the last still runs,
// never changes what an earlier dispatch reads
// (tests/gpu/argument_table_test.cpp shows it on this machine). The ring and
// the table are made once, at construction (MEM.9).
//
// Cost of recording a frame, on the CPU: 96 bytes written to the ring (32 of
// frame constants and, with a camera, 64 of camera), one framing of the camera
// (a few dozen flops), one compute encoder, and per pass one pipeline bind,
// its argument-table entries and one dispatch, all in one command buffer
// (GPU.6). When shapes move, also: each moving shape's motion, its
// transform's write and its box's (core/animation/animate.h,
// scene_acceleration.h), and one structure build and one barrier.
// Nothing is allocated.
//
// ReSTIR DI. A graph with its passes (core/frame/schedule.h) gets, at
// construction, everything below and nothing else changes for a graph
// without them. Implemented: restir_initial, restir_temporal in basic mode,
// restir_spatial in either mode (its rays are this frame's) and
// restir_shade. Refused at construction by MetalError, naming what is
// refused, never run without it: a graph with restir_temporal in raytraced
// mode, whose merge traces last
// frame's surface against last frame's structure, which is not yet kept
// (scene_acceleration.h, Keep::last_too). In detail:
//
//   - its passes, each built with where it writes its reservoirs, from the
//     graph's order (frame_resources.h, ReservoirOutput): the initial pass
//     to the working array when a merge follows, else the final one; the
//     temporal merge in place when a spatial merge follows, else to the
//     final one; the spatial merge always to the final one;
//   - ReuseImages (frame_images.h), made by prepare() at the frame's size
//     and asked by it, from the frame's inputs' reused_since
//     (core/frame/frame_inputs.h), whether the frame has a previous one;
//   - a second ring beside the frame constants', of the reuse constants
//     (core/contracts/reuse_constants.h), 128 bytes a frame: the settings,
//     whether there is a previous frame, and the previous frame's camera,
//     framed for this frame's size, which the renderer keeps from the frame
//     before (its inputs' camera: the renderer holds no camera of its own
//     beyond the one it was last given);
//   - for a graph with a temporal merge: last frame's transforms and glows,
//     a second ShapeTransforms and LightGlows, which recording a frame fills
//     by animating at the previous frame's time, kept from the frame before
//     as its camera is (metal/scene/shape_transforms.h, light_glows.h), and
//     only what moves or glows, as for this frame's; and, in raytraced
//     mode only (core/contracts/reuse_settings.h), the acceleration
//     structure kept a frame longer (scene_acceleration.h, Keep::last_too), which basic
//     mode's temporal merge, tracing no ray, never reads. At the first
//     frame of a history there is no previous frame, and none of it is
//     read;
//   - the scene's media counted: a scene with 0xffff media or more is
//     refused by MetalError, since the path state keeps a medium's index in
//     16 bits (core/contracts/surface_record.h; I.5);
//   - the scene's materials checked against the packed Bsdf's range: a
//     scene with a color or scalar a material resolves to that is not
//     finite or is above 65504 in magnitude is refused by MetalError naming
//     the material and the value, since the surface record packs them as
//     halves (core/contracts/bsdf.h; core/scene/packed_range.h; I.5).
//
// Barriers, each naming its hazard (GPU.8): the queue barrier before the
// initial pass, the first that writes what frames share (this frame's
// halves of the ping-pongs, which the frame before read as its own, and
// the accumulated image, through the shade); and a dispatch-to-dispatch
// barrier before each later ReSTIR pass, each reading what the one before
// wrote, and before the presenting pass, which reads the shade's
// accumulated image. Four dispatches and four barriers for the light, where
// the path graph has one dispatch and one barrier.
//
// The argument table of a ReSTIR graph holds 16 buffer bindings, the
// passes' most being 12 (passes/bindings.h); every other graph's holds the
// 8 it did.
//
// ReSTIR PT. A graph with its passes gets, at construction, its passes;
// PathReuseImages (frame_images.h), made by prepare() at the frame's size
// and, with restir_pt_temporal, asked whether the frame has a previous one
// (frame::ReuseHistory, as ReSTIR DI's arrays are); one static buffer of its
// settings and, with restir_pt_spatial, the three reuse textures, built
// once on the CPU from the graph's sigma and a fixed seed
// (core/sampling/pairing.h; 323 KB); and with restir_pt_temporal, a ring of
// its reuse constants (contracts/path_reuse_settings.h, 80 bytes a frame:
// the previous camera, whether there is a previous frame) and last frame's
// transforms and glows, as ReSTIR DI's temporal merge has them. Refused at
// construction by MetalError, as the G-buffer and the light tiles pack
// them (I.5): a scene with 0xffff media or more, a material value outside
// the packed BSDF's range, more than 65536 shapes or lights. Barriers as
// ReSTIR DI's: the queue barrier before the initial pass, a dispatch
// barrier before each later ReSTIR PT pass and, inside the passes, between
// their dispatches (metal/passes/restir_pt_*/). Its argument table holds 26
// bindings, its passes' most being 26 (passes/bindings.h).
//
// The features pass (contract 17, core/contracts/features.h). A graph with
// it gets, at construction, its pass; FeatureImages (feature_images.h),
// made by prepare() at the frame's size; a ring of its constants, 80 bytes
// a frame: whether the frame has one before it (its inputs' previous,
// core/frame/frame_inputs.h) and that frame's camera framed for this
// frame's size; and, for a scene whose shapes move, a third ShapeTransforms,
// which recording a frame fills by animating what moves at the previous
// frame's time, the motion's other end (a still scene binds this frame's).
// Apart from a temporal graph's last transforms, which are at the time the
// renderer last recorded, a sample of this frame's own instant when a
// frame is several (frame_inputs.h, previous). Nothing else changes for a
// graph without it. Its images are shared by the frames in flight, so the
// pass counts among those the queue barrier waits before; no barrier of its
// own in the frame: its one reader, the denoise pass, comes after the
// barrier recorded for the radiance that pass reads. read_features() copies
// them back.
//
// The denoise pass (core/frame/schedule.h; passes/denoise/denoise.h). A
// graph with it renders at two sizes (frame::light_size): every pass before
// it at the light size, half the target's each way, rounded up, and the
// denoise and presenting passes at the target's. At construction it gets
// the network, read from the graph's weights (core/denoise/network.h);
// DenoiseImages, made by prepare() at the light size and asked whether the
// frame has a history (denoise_images.h); a second ring of frame constants
// and camera, framed at the target's size, beside the first, which is then
// framed at the light size; a ring of its constants (contract 18, 80
// bytes a frame: the camera's view axes, the sizes, whether there is a
// history); and the full-size albedo it remodulates by: a second
// FeatureImages at the target's size and a second ring of feature
// constants, written by the features kernel dispatched again at the
// target's size, as the training's full-size albedo was rendered (docs/
// research/2026-10-14-denoiser-features.md), recorded as part of the
// denoise step. Its accumulated image, ReSTIR's arrays, MNEE's list and
// the features are at the light size; the radiance image and the bloom
// pyramid at the target's, the radiance image the denoise pass's output,
// not the accumulated image. Its argument table holds 8 textures. A graph
// without it renders as it did: one size, one ring, the same bindings.
//
// Cost of recording a ReSTIR frame, on the CPU, beyond the above: 128 bytes
// of reuse constants, four pipeline binds and dispatches with their
// argument-table entries (two of each without the merges); with a temporal merge, one more animation of
// what moves and glows, at the previous time (O(movers + glowers), as
// animate()'s cost is: core/animation/animate.h), and no
// second structure build: last frame's structure is the one last frame
// built. Nothing is allocated.
class Renderer {
public:
    // Builds a pass for each entry of `schedule` from the backend's shader
    // library, compiled into the program (cmake/MetalLibrary.cmake), makes the
    // ring resident through `submission`, and, if any pass reads the scene,
    // puts `scene` on the GPU. `scene` may be null when no pass reads it.
    // Throws MetalError if the schedule is invalid (frame::invalid), a pass needs a
    // scene and there is none, a glowing light is not one of the scene's, a
    // ReSTIR graph's scene has more media or a wider material value than its
    // path state and surface record hold (above), or a pipeline, buffer or
    // structure cannot be built.
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
    // (frame_images.h), are made at `size`; and, for a graph with the
    // features pass, the frame before, if given, must have a finite time and
    // a camera that can be framed. Called before the frame's submission
    // begins, for every graph; throws MetalError if any fails.
    //
    // Everything a frame's inputs or the scene can get wrong is checked here
    // or at construction, before the submission begins (E.4): an exception
    // after begin() leaves the submission open, never to be committed
    // (submission.h), so record() throws only for a broken protocol or a
    // Metal failure, which end the run.
    void prepare(const frame::FrameInputs& inputs, frame::Extent size);

    // Records frame `inputs` into `begun`, writing `target`, of `size`.
    // `begun` is what Submission::begin() returned. Throws MetalError if the
    // frame was not prepared at this size, if the graph uses images or
    // accumulates; if it has no camera and the graph reads a scene; or if
    // Metal makes no encoder.
    void record(const FrameSlot& begun, const frame::FrameInputs& inputs, MTL::Texture* target,
                frame::Extent size);

    // Samples left out for not being finite (metal/film/non_finite.h), over
    // every completed frame; 0 for a graph that accumulates nothing. Reads
    // no counter a frame in flight may be writing.
    std::uint64_t non_finite_samples() const noexcept;

    // The accumulated image's floats into `out`, RGBA a pixel, rows top to
    // bottom: what the passes read, at full precision, for a test or tool
    // that compares images below what 8 bits a channel can show. In a graph
    // with the denoise pass, its radiance image instead: what the
    // presenting pass reads, at the target's size. A readback
    // (GPU.1), not for a frame loop: it waits for its copy. The frames it
    // reads must have completed (Submission::wait_until_complete). Throws
    // MetalError if the graph has no accumulated image, no frame has made it
    // yet, or `out` is not its size.
    void read_accumulated(std::span<float> out);

    // The feature images (contract 17) as the last frame wrote them, read
    // back (feature_images.h, FeatureImages::read): a readback (GPU.1), not
    // for a frame loop. The frames it reads must have completed. Throws
    // MetalError if the graph has no features pass or no frame has made its
    // images yet.
    FeatureImages::Read read_features() const;

private:
    using Pass = std::variant<TestPatternPass, PreviewPass, PathPass, DisplayPass, ToneMapPass, RestirInitialPass,
                              RestirTemporalPass, RestirSpatialPass, RestirShadePass, RestirPtInitialPass,
                              RestirPtTemporalPass, RestirPtSpatialPass, RestirPtShadePass,
                              RestirPtDuplicationPass, FeaturesPass, DenoisePass>;

    // A pass of the schedule, with its kind (C.1).
    struct Step {
        frame::PassKind kind;
        Pass pass;
    };

    // The frame prepare() readied, until it is recorded.
    struct Prepared {
        std::uint64_t index = 0;
        frame::Extent size;
        std::uint32_t accumulated_frames = 0;
        std::uint64_t reused_frames = 0;  // the reuse history's frames before this one
        bool denoise_history = false;     // the denoiser reads last frame's output
    };

    Submission& submission_;
    Library library_;
    FrameArray constants_;  // the ring: one slot per frame in flight
    NS::SharedPtr<MTL4::ArgumentTable> arguments_;  // every frame's: Metal snapshots it at each dispatch
    bool needs_scene_ = false;  // some pass in the schedule reads the scene
    std::unique_ptr<SceneBuffers> scene_;
    std::unique_ptr<SceneAcceleration> acceleration_;
    std::unique_ptr<ShapeTransforms> transforms_;
    std::unique_ptr<LightGlows> glows_;
    animation::Animation animation_;  // empty unless the scene changes
    std::unique_ptr<FrameImages> images_;         // when a pass writes radiance, unless into the accumulated image
    std::unique_ptr<Accumulation> accumulation_;  // when a pass accumulates
    std::unique_ptr<NonFinite> non_finite_;       // with it
    // ReSTIR DI's (above): its settings, as the graph gives them; its
    // arrays; and the ring of its reuse constants, one copy per frame in
    // flight. None for a graph without its passes.
    std::optional<passes::RestirDi> restir_;
    // ReSTIR PT's (above): its arrays, and its settings and reuse textures.
    // None for a graph without its passes.
    std::unique_ptr<PathReuseImages> path_reuse_;
    std::unique_ptr<MneeList> mnee_list_;  // with mnee: MNEE's work list (contract 16)
    std::unique_ptr<StaticArrays> path_static_;
    std::unique_ptr<FrameArray> path_constants_;  // with restir_pt_temporal: contracts::PathReuseConstants
    std::unique_ptr<ReuseImages> reuse_;
    std::unique_ptr<FrameArray> reuse_constants_;
    // The features pass's (above): its images, its constants' ring, and the
    // shapes as the previous frame placed them, when shapes move. None
    // without the pass.
    std::unique_ptr<FeatureImages> features_;
    std::unique_ptr<FrameArray> feature_constants_;
    std::unique_ptr<ShapeTransforms> feature_transforms_;
    // The denoise pass's (above): the light size's rule, its buffers, the
    // target-size ring, its constants' ring, and the full-size albedo's
    // images, constants and kernel. None without the pass.
    bool denoises_ = false;
    frame::Schedule schedule_;  // for light_size()
    std::uint32_t denoise_features_ = 0;  // the network's last width
    std::unique_ptr<DenoiseImages> denoise_images_;
    std::unique_ptr<FrameArray> output_constants_;
    std::unique_ptr<FrameArray> denoise_constants_;
    std::unique_ptr<FeatureImages> full_features_;
    std::unique_ptr<FrameArray> full_feature_constants_;
    std::unique_ptr<FeaturesPass> full_features_pass_;
    // A temporal graph's last frame (above): its transforms and glows,
    // animated at its time each frame, only when the scene changes (a still
    // scene's one copy is last frame's too); and the camera and time the
    // frame before was recorded with, none before the first.
    bool reuses_history_ = false;
    std::unique_ptr<ShapeTransforms> last_transforms_;
    std::unique_ptr<LightGlows> last_glows_;
    std::optional<contracts::Camera> previous_camera_;
    frame::Seconds previous_time_{0.0};
    std::optional<Prepared> prepared_;
    std::vector<Step> steps_;  // in schedule order
    // The first pass that writes the images between passes, accumulates or
    // is one of ReSTIR DI's (whose arrays the frames share), before which
    // the queue barrier is recorded; none if none does.
    std::optional<std::size_t> first_cross_frame_;
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
// render_to_window() refuses, by MetalError, a drawable whose texture is not the
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
