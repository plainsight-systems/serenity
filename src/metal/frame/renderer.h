#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "core/animation/animate.h"
#include "core/contracts/camera.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/denoise/network.h"
#include "core/frame/schedule.h"
#include "core/passes/restir_di.h"
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
#include "metal/film/failure_count.h"
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
// failure_count.h), which non_finite_samples() reads. A frame whose inputs
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
// the drawable orders the frame against the display (submission.h). A
// denoised frame's two halves each have such a barrier, on their own
// queues (below, "Two halves").
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
// mode, whose merge traces last frame's surface against last frame's
// structure, which is not kept (scene_acceleration.h; its design waits in
// docs/process/QUEUE.md). In detail:
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
//     only what moves or glows, as for this frame's (basic mode's merge
//     traces no ray, so needs no structure of last frame's). At the first
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
// The argument table of a ReSTIR graph holds its passes' most buffer
// bindings (passes/bindings.h), as every graph's holds its families' most
// (renderer.cpp, buffer_bindings(), each family's count asserted against
// its bindings there).
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
// their dispatches (metal/passes/restir_pt_*/). Its argument table holds
// its passes' most bindings, as ReSTIR DI's does.
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
// graph without it. In a denoised graph its images are one set per frame
// slot, the hand-off (below, "Two halves"); in any other, one set the
// frames in flight share, so the
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
// research/2026-10-14-denoiser-features.md), recorded at the end of the
// light half (below, "Two halves"). Its accumulated image, ReSTIR's
// arrays, MNEE's list and the features are at the light size; the
// radiance image and the bloom pyramid at the target's, the radiance image
// the denoise pass's output, not the accumulated image. Its argument table
// holds the denoiser's textures (renderer.cpp, texture_bindings()). A
// graph without it renders as it did: one size, one ring, the same
// bindings.
//
// Two halves (logical-overview.md, "The frame"; Milestone 4, docs/research/
// 2026-10-17-milestone-4-plan.md, Step 1). A denoised frame splits at the
// denoise pass: the light half, animate, every pass before it and the
// full-size albedo's features kernel, bound by the ray-tracing unit; the
// image half, the denoise pass and every pass after it, bound by
// arithmetic. The full-size albedo is the denoiser's input, but its kernel
// traces rays: in the image half it ran beside the light half's tracing,
// both on the ray-tracing unit, and took 32 ms there against 8.5 alone at
// the window's size; in the light half it runs beside the network, and the
// frame is shown sooner (the note, Step 1, measures both). parts() is
// Parts::two for such a graph and Parts::one for any other, and
// render_to_window() and render_to_offscreen() begin each frame's
// submission with it (submission.h, "Two parts"): the light half in the
// first part, on the main queue, the image half in the second, on the
// overlap queue, so frame N's image half may run beside frame N + 1's
// light half. What orders them is the submission's: part 2 after part 1 of
// the same frame by its event; each half after its own half of the frame
// before by its queue and the queue barrier recorded at its start, so each
// half's history (the light's reservoirs and accumulated image; the
// denoiser's last output, the radiance image and bloom) passes from frame
// to frame in order, as before.
//
// The hand-off: what the image half reads of what the light half wrote.
// The invariant: frame N + 1's light half never writes what frame N's image
// half reads (CP.3: never one copy shared by a writer and a reader at
// once). Everything the image half reads of the light half's is ringed by
// the frame slot, as the constants are (GPU.7, MEM.4: a resource per frame
// in flight; submission.h: a slot is reused only once its frame, image
// half included, has completed):
//   - the features, written whole by the features pass each frame and read
//     by nothing else on the light's queue: a FeatureImages per slot, the
//     features pass writing the frame's slot's;
//   - the light's radiance: the accumulated image carries the light's
//     history across frames (a still scene's mean, accumulation.h), so it
//     stays one image on the light's queue, and the light half ends by
//     copying it into the slot's hand-off image (denoise_images.h,
//     DenoiseImages::light_radiance), which the denoise pass reads in its
//     place: RGBA32Float at the light size, 31 MB at 1728 x 1117, copied
//     once a frame, about 0.1 ms of the light half (measured in the note);
//   - the full-size albedo: the features kernel at the output's size writes
//     all four of its images whole each frame, the albedo the one the
//     denoiser reads, so, as the light-size features, a FeatureImages at
//     the output's size per slot;
//   - the slot's constants, which the image half reads: ringed already.
// What the image half writes the light half never reads. Memory, beyond a
// graph of one part: a second set of features at the light size (28 bytes
// a pixel, 54 MB at 1728 x 1117), a second at the output's size (216 MB at
// 3456 x 2234) and two radiance copies (16 bytes a pixel, 62 MB): 332 MB,
// made at prepare(), never per frame (GPU.9). Copying only the albedo out
// of one set would save 216 MB for 62 MB of copies on the light's queue
// every frame; the light half is the longer, so the memory is spent.
//
// Barriers (GPU.8), each naming its hazard: the light half's queue barrier
// before its first pass that crosses frames waits for the light queue's
// copies as well as its dispatches (dispatch and blit before dispatch),
// since its shade writes the accumulated image the frame before's copy
// read; a barrier from the shade's dispatch, and the full-size albedo's,
// to the copy (dispatch before blit), which in one part also orders the
// albedo before the image half. The image half's queue barrier, at its
// start, waits for the overlap queue's dispatches before it: the frame
// before's image half, which read and wrote what this one reads and
// writes (the history, the radiance image, the bloom). The light half's
// writes reach the image half through the submission's event.
//
// One part for a denoised graph. A frame recorded into a submission begun
// with one part (a test's, which begins its own) records both halves into
// the first part's one encoder, the copy between them, and a barrier from
// the copy to the image half (blit before dispatch) in place of the event:
// the same work in the same order on one queue, so the same pixels. The
// image half then records no queue barrier of its own: the light half's
// already waited for the frame before, its image half included. A
// submission of two parts for a graph of one is refused by MetalError.
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

    // How many parts a frame's submission has (above, "Two halves"): two
    // for a denoised graph, one for any other.
    Parts parts() const noexcept { return denoises_ ? Parts::two : Parts::one; }

    // Records frame `inputs` into `begun`, writing `target`, of `size`.
    // `begun` is what Submission::begin() returned, of parts() or of one
    // part (above, "One part for a denoised graph"). Throws MetalError if
    // the frame was not prepared at this size, if the graph uses images or
    // accumulates; if it has no camera and the graph reads a scene; if
    // `begun` has two parts and the graph one; or if Metal makes no
    // encoder.
    void record(const FrameSlot& begun, const frame::FrameInputs& inputs, MTL::Texture* target,
                frame::Extent size);

    // Samples left out for not being finite (metal/film/failure_count.h),
    // over every completed frame; 0 for a graph that accumulates nothing.
    // Reads no counter a frame in flight may be writing.
    std::uint64_t non_finite_samples() const noexcept;

    // MNEE requests the work list had no room for, their light lost
    // (contract 16), over every completed frame, as non_finite_samples()
    // counts; 0 for a graph without mnee.
    std::uint64_t mnee_lost_requests() const noexcept;

    // The accumulated image's floats into `out`, RGBA a pixel, rows top to
    // bottom: what the passes read, at full precision, for a test or tool
    // that compares images below what 8 bits a channel can show. In a graph
    // with the denoise pass, its radiance image instead: what the
    // presenting pass reads, at the target's size. A readback
    // (GPU.1), not for a frame loop: it waits for every frame committed
    // before it and then for its copy (metal/device/readback.h). Throws
    // MetalError if the graph has no accumulated image, no frame has made it
    // yet, or `out` is not its size.
    void read_accumulated(std::span<float> out);

    // The feature images (contract 17) as the last frame wrote them, read
    // back (feature_images.h, FeatureImages::read): a readback (GPU.1), not
    // for a frame loop, as read_accumulated() is. Throws
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

    // Each family's state (above), one struct each, held in place in an
    // optional, present only for a graph with the family's passes (C.1,
    // R.5). Each binds its share of a frame's resources.
    //
    // ReSTIR DI's: its settings, as the graph gives them; its arrays; and
    // the ring of its reuse constants, one copy per frame in flight.
    struct RestirDiFamily {
        RestirDiFamily(const Device& device, Submission& submission, const passes::RestirDi& graph_settings,
                       bool temporal, bool spatial);
        // `previous_camera` last frame's where the frame merges it, else none.
        void bind(FrameResources& resources, const FrameSlot& begun, std::uint64_t index,
                  const std::optional<contracts::Camera>& previous_camera, frame::Extent light);

        passes::RestirDi settings;
        ReuseImages images;
        FrameArray constants;
    };
    // ReSTIR PT's: its arrays, its settings and reuse textures, and with
    // temporal reuse the ring of its reuse constants.
    struct RestirPtFamily {
        RestirPtFamily(const Device& device, Submission& submission, const contracts::PathReuseSettings& settings,
                       std::span<const std::uint16_t> textures, bool spatial, bool temporal);
        void bind(FrameResources& resources, const FrameSlot& begun, std::uint64_t index,
                  const std::optional<contracts::Camera>& previous_camera, frame::Extent light);

        PathReuseImages images;
        StaticArrays statics;                // the settings, then the reuse textures
        std::optional<FrameArray> constants;  // with restir_pt_temporal: contracts::PathReuseConstants
    };
    // MNEE's (contract 16): its work list and its count of lost requests.
    struct MneeFamily {
        MneeFamily(const Device& device, Submission& submission, MneeList::Light light);
        void bind(FrameResources& resources, const FrameSlot& begun);

        MneeList list;
        FailureCount lost;
    };
    // The features pass's: its images, one set or, handed off, one per
    // frame slot (above, "Two halves"); its constants' ring; and the shapes
    // as the previous frame placed them, when shapes move.
    struct FeatureFamily {
        enum class Sets { one, per_slot };
        FeatureFamily(const Device& device, Submission& submission, const scene::SceneDescription& scene,
                      bool moves, Sets sets);
        void prepare(frame::Extent light);
        // The set frame slot `slot`'s frame writes.
        const FeatureImages& images(std::uint32_t slot) const;
        void bind(FrameResources& resources, const FrameSlot& begun,
                  const std::optional<frame::PreviousFrame>& previous, frame::Extent light);

        std::array<std::optional<FeatureImages>, frames_in_flight> sets;  // the second only per slot
        FrameArray constants;
        std::optional<ShapeTransforms> previous_transforms;
        std::uint32_t last_slot = 0;  // the slot of the frame bound last, which read_features() reads
    };
    // The denoise pass's: the network's last width, its buffers, the
    // target-size ring, its constants' ring, and the full-size albedo's
    // images, constants and kernel.
    struct DenoiseFamily {
        DenoiseFamily(const Device& device, Submission& submission, const Library& library,
                      const denoise::Network& network);

        std::uint32_t features;
        DenoiseImages images;
        FrameArray output_constants;
        FrameArray constants;
        std::array<std::optional<FeatureImages>, frames_in_flight> full_features;  // a set per slot (above)
        FrameArray full_feature_constants;
        FeaturesPass full_features_pass;
    };
    // A denoised graph's resources from the denoise pass on (renderer.cpp,
    // denoise_resources()); every pass's own without the pass.
    struct DenoiseResources {
        FrameResources output;
        FrameResources denoising;
        FrameResources full_features;
    };

    // The constructor's parts (F.3), in its order.
    void make_steps(const Device& device, const frame::Schedule& schedule,
                    const std::optional<denoise::Network>& network);
    void make_images(const Device& device, const frame::Schedule& schedule);
    void make_restir_di(const Device& device, const frame::Schedule& schedule, const scene::SceneDescription* scene);
    void make_restir_pt(const Device& device, const frame::Schedule& schedule, const scene::SceneDescription* scene);
    void make_mnee(const Device& device, const frame::Schedule& schedule, const scene::SceneDescription* scene);
    void make_scene(const Device& device, const scene::SceneDescription& scene);

    // Whether the graph has anything prepare() readies, so record() asks
    // that it was: every member prepare() prepares (I.5; one test, so the
    // two cannot disagree).
    bool prepares() const noexcept;

    // record()'s parts (F.3), in its order.
    Prepared take_prepared(const frame::FrameInputs& inputs, frame::Extent size);
    static std::pair<MTL::GPUAddress, MTL::GPUAddress> write_constants(FrameArray& ring, const FrameSlot& begun,
                                                                       const frame::FrameInputs& inputs,
                                                                       frame::Extent at,
                                                                       std::uint32_t accumulated_frames);
    FrameResources frame_resources(const FrameSlot& begun, const frame::FrameInputs& inputs, MTL::Texture* target,
                                   frame::Extent light, const Prepared& prepared, bool previous);
    void animate(MTL4::ComputeCommandEncoder* encoder, const FrameSlot& begun, const frame::FrameInputs& inputs,
                 bool previous);
    DenoiseResources denoise_resources(const FrameResources& resources, const FrameSlot& begun,
                                       const frame::FrameInputs& inputs, frame::Extent size,
                                       const Prepared& prepared);
    // The light half (every step before the denoise pass; all of them in a
    // graph without it), then the full-size albedo and the hand-off's copy;
    // and the image half, from the denoise pass on (above, "Two halves").
    void record_light(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources);
    void record_hand_off(MTL4::ComputeCommandEncoder* encoder, const FrameSlot& begun);
    void record_image(MTL4::ComputeCommandEncoder* encoder, const DenoiseResources& denoised_resources);

    Submission& submission_;
    Library library_;
    FrameArray constants_;  // the ring: one slot per frame in flight
    NS::SharedPtr<MTL4::ArgumentTable> arguments_;  // every frame's: Metal snapshots it at each dispatch
    bool denoises_ = false;
    frame::Schedule schedule_;  // for light_size()
    bool needs_scene_ = false;  // some pass in the schedule reads the scene
    std::unique_ptr<SceneBuffers> scene_;
    std::unique_ptr<SceneAcceleration> acceleration_;
    std::unique_ptr<ShapeTransforms> transforms_;
    std::unique_ptr<LightGlows> glows_;
    animation::Animation animation_;  // empty unless the scene changes
    std::unique_ptr<FrameImages> images_;         // when a pass writes radiance, unless into the accumulated image
    std::unique_ptr<Accumulation> accumulation_;  // when a pass accumulates
    std::unique_ptr<FailureCount> non_finite_;    // with it: samples not finite
    std::optional<RestirDiFamily> restir_di_;
    std::optional<RestirPtFamily> restir_pt_;
    std::optional<MneeFamily> mnee_;
    std::optional<FeatureFamily> features_;
    std::optional<DenoiseFamily> denoise_;
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
    // The denoise pass's step, where the image half begins; steps_.size()
    // in a graph without it, all of whose steps are the light half's.
    std::size_t image_begins_ = 0;
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
