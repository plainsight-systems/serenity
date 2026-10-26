#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "core/animation/animate.h"
#include "core/contracts/camera.h"
#include "core/denoise/network.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/sampling/jitter.h"
#include "core/scene/scene.h"
#include "metal/acceleration/scene_acceleration.h"
#include "metal/device/device.h"
#include "metal/device/frame_array.h"
#include "metal/device/library.h"
#include "metal/device/offscreen.h"
#include "metal/device/presenter.h"
#include "metal/device/static_arrays.h"
#include "metal/device/submission.h"
#include "metal/film/failure_count.h"
#include "metal/frame/accumulation.h"
#include "metal/frame/checkerboard_images.h"
#include "metal/frame/denoise_images.h"
#include "metal/frame/depth_of_field_images.h"
#include "metal/frame/feature_images.h"
#include "metal/frame/frame_images.h"
#include "metal/frame/frame_resources.h"
#include "metal/frame/mnee_list.h"
#include "metal/passes/checkerboard_fill/checkerboard_fill.h"
#include "metal/passes/denoise/denoise.h"
#include "metal/passes/depth_of_field/depth_of_field.h"
#include "metal/passes/features/features.h"
#include "metal/passes/path/path.h"
#include "metal/passes/restir_pt_duplication/restir_pt_duplication.h"
#include "metal/passes/restir_pt_initial/restir_pt_initial.h"
#include "metal/passes/restir_pt_shade/restir_pt_shade.h"
#include "metal/passes/restir_pt_spatial/restir_pt_spatial.h"
#include "metal/passes/restir_pt_temporal/restir_pt_temporal.h"
#include "metal/passes/test_pattern/test_pattern.h"
#include "metal/passes/tone_map/tone_map.h"
#include "metal/scene/light_glows.h"
#include "metal/scene/scene_buffers.h"
#include "metal/scene/shape_transforms.h"

namespace serenity::metal {

// The shade's step in a graph with the spatial pass, whose merge shades
// (RTXDI's fused shading: passes/restir_pt_spatial/restir_pt_spatial.h):
// the schedule puts the shade straight after it, so the step records no
// dispatch and no barrier (Renderer's record_light() skips it).
struct ShadedInMerge {};

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
// refuses, by MetalError, a graph that needs a scene when given none. Each
// frame then gives every pass the same resources (frame_resources.h), the scene
// among them. A graph that reads no scene runs with or without one; given one,
// it is not copied, and the structure is built over the scene's shapes as the
// Shape family bounds them (shapes/shapes.h).
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
// renderer applies at construction, refusing by MetalError any schedule the
// core would not accept. Each frame, prepare() readies it from the frame's
// inputs before the frame's submission begins, since remaking it for a new size
// waits for the frames in flight; record() then gives the pass the image and
// the count of frames it holds, through the frame constants, and Film's
// counter of samples left out for not being finite (metal/film/
// failure_count.h), which non_finite_samples() reads. A frame whose inputs
// claim an image the history does not hold is refused, by MetalError, and so is
// a frame recorded without being prepared, and, when the scene changes, a frame
// at another time than the frames the image holds (accumulation.h).
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
// is one of ReSTIR PT's (whose arrays, PathReuseImages, the frames share too),
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
// ReSTIR PT. A graph with its passes gets, at construction, its passes,
// their pipelines specialized by the run's settings (passes/restir_pt/
// reuse_constants.h); PathReuseImages (frame_images.h), made by prepare() at
// the frame's size and, with restir_pt_temporal, asked from the frame's inputs'
// reused_since (core/frame/frame_inputs.h, frame::ReuseHistory) whether the
// frame has a previous one; one static buffer of, with restir_pt_spatial, the
// three reuse textures (one zero texel without), built once on the CPU from the
// graph's sigma and a fixed seed (core/sampling/pairing.h; 323 KB); and with
// restir_pt_temporal, a ring of its reuse constants
// (contracts/path_reuse_settings.h, 80 bytes a frame: the previous camera,
// whether there is a previous frame, which the renderer keeps from the frame
// before: its inputs' camera, framed for this frame's size) and last frame's
// transforms and glows, a second ShapeTransforms and LightGlows, which
// recording a frame fills by animating what moves or glows at the previous
// frame's time (metal/scene/ shape_transforms.h, light_glows.h); at the first
// frame of a history there is no previous frame, and none of it is read.
// Refused at construction by MetalError, as the G-buffer and the light tiles
// pack them (I.5): a scene with 0xffff media or more, a material value outside
// the packed BSDF's range, more than 65536 shapes or lights. Barriers, each
// naming its hazard (GPU.8): the queue barrier before the initial pass, the
// first that writes what frames share; a dispatch barrier before each
// later ReSTIR PT pass, each reading what the one before wrote, and,
// inside the passes, between their dispatches (metal/passes/restir_pt_*/).
// Its argument table holds its passes' most buffer bindings
// (passes/bindings.h), as every graph's holds its families' most
// (renderer.cpp, buffer_bindings(), each family's count asserted against
// its bindings there).
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
//     stays one image on the light's queue, and the light half ends with
//     the checkerboard's fill (below), which writes the slot's hand-off
//     image whole from it (denoise_images.h, DenoiseImages::light_radiance),
//     and the denoise pass reads the hand-off in its place: RGBA32Float at
//     the light size, 31 MB at 1728 x 1117;
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
// dispatches, the frame before's fill among them, which read the
// accumulated image this frame's shade writes; a barrier from the shade's
// dispatch, the features' and the full-size albedo's, to the fill
// (dispatch before dispatch), which in one part also orders the albedo
// before the image half. The image half's queue barrier, at its
// start, waits for the overlap queue's dispatches before it: the frame
// before's image half, which read and wrote what this one reads and
// writes (the history, the radiance image, the bloom). The light half's
// writes reach the image half through the submission's event.
//
// One part for a denoised graph. A frame recorded into a submission begun
// with one part (a test's, which begins its own) records both halves into
// the first part's one encoder, the fill between them, and a barrier from
// the fill to the image half (dispatch before dispatch) in place of the
// event:
// the same work in the same order on one queue, so the same pixels. The
// image half then records no queue barrier of its own: the light half's
// already waited for the frame before, its image half included. A
// submission of two parts for a graph of one is refused by MetalError.
//
// Checkerboard rendering (contract 20, core/contracts/checkerboard.h;
// logical-overview.md, "The frame" and principle 13; Milestone 4's Step 4):
// how a graph whose light is the network's input (core/frame/schedule.h,
// feeds_network: ReSTIR PT with the features pass, the denoised graph and
// the data set's input graph alike) traces its light.
// It gets, at construction, ReSTIR PT's settings with checkerboard 1, its
// reuse textures built at sigma over the traced lattice's step
// (metal/reuse/pairing.metal.h, lattice_partner), the fill
// (passes/checkerboard_fill/checkerboard_fill.h), a ring of its
// constants, 16 bytes a frame, and its history (checkerboard_images.h),
// made by prepare() at the light size. Each frame, ReSTIR PT's passes are
// given the packed grid to dispatch over (frame_resources.h, traced), and
// the light half ends with a barrier from the dispatches before it (the
// shade's accumulated image, the features' images and the full-size
// albedo's: GPU.8, a write and its reader), then the fill, writing the
// slot's hand-off whole; in a graph without the denoise pass the fill comes
// before the presenting pass instead, after the same barrier, and writes the
// fill's own image (checkerboard_images.h, filled), which the presenting
// pass shows (the barrier before it, as before every pass that reads the
// radiance) and read_accumulated() reads back, the light the headless
// renderer writes: what a denoised graph hands its network, to the bit
// (tests/gpu/features_test.cpp). The fill reads the frame before's history,
// by the frame's parity, and writes
// this frame's, the image frame n - 1 read, which the light half's queue
// barrier orders after that read. It is told the history holds this
// history's frame before (contract 20's CheckerboardConstants, history 1)
// when its own reuse history does (checkerboard_images.h: this history,
// this size, the frame before), which the parity then holds whatever was
// submitted between the two frames. Any other graph traces every pixel and
// has none of it. Memory: the fill's history, 62 MB at 1728 x 1117, and
// without the denoise pass its own image, 31 MB more; the packed arrays use
// the first half of the full-size ones. Its argument table holds the
// fill's textures, as a denoised graph's does.
//
// The depth of field (core/frame/schedule.h, depth_of_field; passes/
// depth_of_field/depth_of_field.h; logical-overview.md, principle 12), a
// denoised graph's. Such a graph traces through a pinhole: record() takes
// the frame's inputs as frame::traced_inputs() gives them, the camera and
// the frame before's with their lens taken off, for everything it frames
// and binds, the light half's rings, the full-size albedo's, the temporal
// reuse's previous camera and the features' (so prepare() checks the
// cameras the frame was given, which a lens's removal leaves valid); and
// the depth of field's constants alone from the frame's own camera, its
// lens on, at the output's size (core/passes/depth_of_field.h), into a
// ring of its own, 48 bytes a frame. At construction it gets its pass and
// DepthOfFieldImages (depth_of_field_images.h), made by prepare() at the
// output's size. The pass is recorded in the image half after the denoise
// pass, reading and writing the radiance image in place, with the slot's
// full-size depth (the full-size albedo's features, traced through the
// same pinhole); the barrier before it is the one recorded before every
// pass that reads the radiance (below, record_image()), and the tone
// map's after it the same. A pinhole's frame (passes::blurs) records
// nothing for it, and the image is the network's. Memory: 77 MB at
// 3456 x 2234 (its images'). Its argument table is a denoised graph's.
//
// The camera's jitter (core/sampling/jitter.h; core/frame/schedule.h,
// jitters), a ReSTIR PT graph's. The frame's offset is written into the
// light's ring of frame constants (contract 7), which every pass before the
// denoise pass reads, so ReSTIR PT's camera rays and the light-size
// features' take it alike; never into the output size's, so a denoised
// graph's full-size albedo keeps each pixel's own point, unjittered as the
// network's output is. The denoise pass's constants carry it too (contract
// 18). A graph of any other kind writes `jittered` 0, and its pixels draw
// what they drew before the jitter, to the bit.
//
// Cost of recording a ReSTIR PT frame, on the CPU, beyond the above: 80
// bytes of reuse constants, its passes' pipeline binds and dispatches with
// their argument-table entries; with temporal reuse, one more animation of
// what moves and glows, at the previous time (O(movers + glowers), as
// animate()'s cost is: core/animation/animate.h), and no second structure
// build: last frame's structure is the one last frame built. Nothing is
// allocated.
class Renderer {
 public:
  // Builds a pass for each entry of `schedule` from the backend's shader
  // library, compiled into the program (cmake/MetalLibrary.cmake), makes the
  // ring resident through `submission`, and, if any pass reads the scene,
  // puts `scene` on the GPU. `scene` may be null when no pass reads it.
  // Throws MetalError if the schedule is invalid (frame::invalid), a pass needs
  // a scene and there is none, a glowing light is not one of the scene's, a
  // ReSTIR PT graph's scene has more media, shapes or lights or a wider
  // material value than its G-buffer and light tiles hold (above), or a
  // pipeline, buffer or structure cannot be built.
  Renderer(const Device& device, Submission& submission,
           const frame::Schedule& schedule,
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
  void record(const FrameSlot& begun, const frame::FrameInputs& inputs,
              MTL::Texture* target, frame::Extent size);

  // Samples left out for not being finite (metal/film/failure_count.h),
  // over every completed frame; 0 for a graph that accumulates nothing.
  // Reads no counter a frame in flight may be writing.
  std::uint64_t non_finite_samples() const noexcept;

  // MNEE requests the work list had no room for, their light lost
  // (contract 16), over every completed frame, as non_finite_samples()
  // counts; 0 for a graph without ReSTIR PT's passes.
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

  // What the last recorded frame hands the network (logical-overview.md,
  // principle 13), read back as read_accumulated() reads: the fill's
  // light, RGBA a pixel at the light size, rows top to bottom (a denoised
  // graph's hand-off; a graph without the denoise pass's filled image,
  // the one it shows and writes), and the camera jitter the frame's rays
  // took and a denoised graph's network was given. A readback (GPU.1), for
  // a test or a tool, not a frame loop. Throws MetalError if the graph's
  // light is not the network's input (core/frame/schedule.h,
  // feeds_network) or no frame has been recorded.
  struct NetworkInput {
    frame::Extent size;
    std::vector<float> rgba;
    sampling::CameraJitter jitter;
  };
  NetworkInput read_network_input();

 private:
  using Pass =
      std::variant<TestPatternPass, PathPass, ToneMapPass, RestirPtInitialPass,
                   RestirPtTemporalPass, RestirPtSpatialPass, RestirPtShadePass,
                   ShadedInMerge, RestirPtDuplicationPass, FeaturesPass,
                   DenoisePass, DepthOfFieldPass>;

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
    // the reuse history's frames before this one
    std::uint64_t reused_frames = 0;
    bool denoise_history = false;  // the denoiser reads last frame's output
    // the checkerboard's fill reads the frame before's light
    bool fill_history = false;
  };

  // Each family's state (above), one struct each, held in place in an
  // optional, present only for a graph with the family's passes (C.1,
  // R.5). Each binds its share of a frame's resources.
  //
  // ReSTIR PT's: its arrays, its settings and reuse textures, and with
  // temporal reuse the ring of its reuse constants.
  struct RestirPtFamily {
    RestirPtFamily(const Device& device, Submission& submission,
                   const contracts::PathReuseSettings& settings,
                   std::span<const std::uint16_t> textures, bool spatial,
                   bool temporal, std::uint32_t lights);
    void bind(FrameResources& resources, const FrameSlot& begun,
              std::uint64_t index,
              const std::optional<contracts::Camera>& previous_camera,
              frame::Extent light);

    PathReuseImages images;
    StaticArrays statics;  // the settings, then the reuse textures
    // with restir_pt_temporal: contracts::PathReuseConstants
    std::optional<FrameArray> constants;
  };
  // MNEE's (contract 16): its work list and its count of lost requests.
  struct MneeFamily {
    MneeFamily(const Device& device, Submission& submission);
    void bind(FrameResources& resources, const FrameSlot& begun);

    MneeList list;
    FailureCount lost;
  };
  // The features pass's: its images, one set or, handed off, one per
  // frame slot (above, "Two halves"); its constants' ring; and the shapes
  // as the previous frame placed them, when shapes move.
  struct FeatureFamily {
    enum class Sets : std::uint8_t { one, per_slot };
    FeatureFamily(const Device& device, Submission& submission,
                  const scene::SceneDescription& scene, bool moves, Sets sets);
    void prepare(frame::Extent light);
    // The set frame slot `slot`'s frame writes.
    const FeatureImages& images(std::uint32_t slot) const;
    void bind(FrameResources& resources, const FrameSlot& begun,
              const std::optional<frame::PreviousFrame>& previous,
              frame::Extent light);

    // the second only per slot
    std::array<std::optional<FeatureImages>, frames_in_flight> sets;
    FrameArray constants;
    std::optional<ShapeTransforms> previous_transforms;
    // the slot of the frame bound last, which read_features() reads
    std::uint32_t last_slot = 0;
  };
  // The denoise pass's: the network's last width, its buffers, the
  // target-size ring, its constants' ring, and the full-size albedo's
  // images, constants and kernel.
  struct DenoiseFamily {
    DenoiseFamily(const Device& device, Submission& submission,
                  const Library& library, const denoise::Network& network);

    std::uint32_t features;
    DenoiseImages images;
    FrameArray output_constants;
    FrameArray constants;
    // a set per slot (above)
    std::array<std::optional<FeatureImages>, frames_in_flight> full_features;
    FrameArray full_feature_constants;
    FeaturesPass full_features_pass;
  };
  // Checkerboard rendering's (above), a denoised graph's: the fill, its
  // constants' ring and its history.
  struct CheckerboardFamily {
    CheckerboardFamily(const Device& device, Submission& submission,
                       const Library& library,
                       CheckerboardImages::Filled filled);

    CheckerboardFillPass fill;
    FrameArray constants;
    CheckerboardImages images;
  };
  // The depth of field's (above): its images and its constants' ring.
  struct DepthOfFieldFamily {
    DepthOfFieldFamily(const Device& device, Submission& submission);

    DepthOfFieldImages images;
    FrameArray constants;
  };
  // A denoised graph's resources from the denoise pass on (renderer.cpp,
  // denoise_resources()); every pass's own without the pass; and whether
  // the depth of field blurs this frame (passes::blurs).
  struct DenoiseResources {
    FrameResources output;
    FrameResources denoising;
    FrameResources full_features;
    FrameResources depth_of_field;
    bool blurs = false;
  };

  // The constructor's parts (F.3), in its order.
  void make_steps(const Device& device, const frame::Schedule& schedule,
                  const std::optional<denoise::Network>& network);
  void make_images(const Device& device, const frame::Schedule& schedule);
  void make_restir_pt(const Device& device, const frame::Schedule& schedule,
                      const scene::SceneDescription* scene);
  void make_mnee(const Device& device, const frame::Schedule& schedule,
                 const scene::SceneDescription* scene);
  void make_scene(const Device& device, const scene::SceneDescription& scene);

  // Whether the graph has anything prepare() readies, so record() asks
  // that it was: every member prepare() prepares (I.5; one test, so the
  // two cannot disagree).
  bool prepares() const noexcept;

  // record()'s parts (F.3), in its order.
  Prepared take_prepared(const frame::FrameInputs& inputs, frame::Extent size);
  // `jitter` the camera's, for a ring whose frames take it (above, "The
  // camera's jitter"), none for one whose do not.
  static std::pair<MTL::GPUAddress, MTL::GPUAddress> write_constants(
      FrameArray& ring, const FrameSlot& begun,
      const frame::FrameInputs& inputs, frame::Extent at,
      std::uint32_t accumulated_frames,
      std::optional<sampling::CameraJitter> jitter);
  FrameResources frame_resources(const FrameSlot& begun,
                                 const frame::FrameInputs& inputs,
                                 MTL::Texture* target, frame::Extent light,
                                 const Prepared& prepared, bool previous);
  // The frame's shapes and glows placed on the host, and last frame's
  // where read; then, in the frame's compute encoder, the structure's
  // update where shapes move (scene_acceleration.h). Apart, so the
  // rasterizer's draws are chosen from the placed shapes before the
  // encoders (passes/features/features.h).
  void animate(const FrameSlot& begun, const frame::FrameInputs& inputs,
               bool previous);
  void update_structure(MTL4::ComputeCommandEncoder* encoder,
                        const FrameSlot& begun);
  // `traced` the frame's inputs as its light is traced, `lens` the
  // frame's own camera, its lens on (above, "The depth of field").
  DenoiseResources denoise_resources(const FrameResources& resources,
                                     const FrameSlot& begun,
                                     const frame::FrameInputs& traced,
                                     const contracts::Camera& lens,
                                     frame::Extent size,
                                     const Prepared& prepared);
  // The light half (every step before the denoise pass; all of them in a
  // graph without it), then the full-size albedo and the checkerboard's
  // fill, which writes the hand-off;
  // and the image half, from the denoise pass on (above, "Two halves").
  // Steps [first, last) of the light half; the whole of a graph without
  // the denoise pass, the fill between its last two in a graph that
  // feeds the network.
  void record_light(MTL4::ComputeCommandEncoder* encoder,
                    const FrameResources& resources, std::size_t first,
                    std::size_t last);
  // The features' raster passes (passes/features/features.h), each its own
  // render encoder in `begun`'s light half before its compute encoder: the
  // light size's, with the features pass, and the full size's, with the
  // denoise pass (`full`, else null). Whether it recorded any, for the
  // compute encoder's barrier on their fragments (GPU.8).
  bool record_raster(const FrameSlot& begun, const FrameResources& resources,
                     const FrameResources* full) const;
  void record_hand_off(MTL4::ComputeCommandEncoder* encoder,
                       const FrameSlot& begun, const FrameResources& resources,
                       const Prepared& prepared, MTL::Texture* hand_off);
  // The image half into `commands`, from `encoder`, which the denoise pass
  // ends and replaces with a new one, left open (passes/denoise/denoise.h).
  void record_image(MTL4::CommandBuffer* commands,
                    MTL4::ComputeCommandEncoder*& encoder,
                    const DenoiseResources& denoised_resources);

  Submission& submission_;
  Library library_;
  FrameArray constants_;  // the ring: one slot per frame in flight
  // every frame's: Metal snapshots it at each dispatch
  NS::SharedPtr<MTL4::ArgumentTable> arguments_;
  bool denoises_ = false;
  // core/frame/schedule.h: traced checkerboarded and filled
  bool feeds_network_ = false;
  frame::Schedule schedule_;  // for light_size()
  bool needs_scene_ = false;  // some pass in the schedule reads the scene
  std::unique_ptr<SceneBuffers> scene_;
  std::unique_ptr<SceneAcceleration> acceleration_;
  std::unique_ptr<ShapeTransforms> transforms_;
  std::unique_ptr<LightGlows> glows_;
  animation::Animation animation_;       // empty unless the scene changes
  std::unique_ptr<FrameImages> images_;  // when a pass writes radiance, unless
                                         // into the accumulated image
  std::unique_ptr<Accumulation> accumulation_;  // when a pass accumulates
  std::unique_ptr<FailureCount> non_finite_;    // with it: samples not finite
  std::optional<RestirPtFamily> restir_pt_;
  std::optional<MneeFamily> mnee_;
  std::optional<FeatureFamily> features_;
  std::optional<DenoiseFamily> denoise_;
  std::optional<CheckerboardFamily> checkerboard_;
  std::optional<DepthOfFieldFamily> depth_of_field_;
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
  // is one of ReSTIR PT's (whose arrays the frames share), before which
  // the queue barrier is recorded; none if none does.
  std::optional<std::size_t> first_cross_frame_;
  // The denoise pass's step, where the image half begins; steps_.size()
  // in a graph without it, all of whose steps are the light half's.
  std::size_t image_begins_ = 0;
  // Where the fill is recorded, before this step: the denoise pass's in a
  // denoised graph, the presenting pass's in another that feeds the
  // network; steps_.size() in any other graph, which has no fill.
  std::size_t fill_at_ = 0;
  // The last recorded frame's slot and the jitter its light took, for
  // read_network_input().
  std::optional<std::uint32_t> last_slot_;
  sampling::CameraJitter last_jitter_{};
};

// One frame, start to finish, for each kind of target. The window and the
// headless renderer call these and nothing below them, so neither names a
// Metal type (file-mapping.md).
//
// Both prepare the frame (Renderer::prepare) before its submission begins.
//
// render_to_window() records the frame into `drawable`, the one the display
// link handed Metal's frame source (metal/frame/display_frames.h), and
// presents it, inside an autorelease pool of its own (what recording gets
// from Metal is autoreleased). It returns the frame's sequence and the
// earlier submission its begin() settled, if any (submission.h), for the
// source to report.
//
// It refuses, by MetalError and before anything is begun, a drawable the
// presenter does not render into (metal/device/presenter.h, renders_into):
// none, or one not its size, which the passes, dispatching a thread per
// pixel of the presenter's size, would write past (I.6). The frame source
// leaves such a drawable before asking for the frame.
//
// render_to_offscreen() records the frame into `target` and commits it,
// without waiting; it returns the submission's sequence, for
// Submission::wait_until_complete() before reading `target` back. It
// drains an autorelease pool of its own too, so a caller with none (the
// headless renderer) leaks nothing a frame. render_offscreen_frame() is
// the same frame, and returns too what its begin() settled, for a caller
// that measures (the headless renderer's --times, headless/options.h).
struct WindowFrame {
  std::uint64_t sequence = 0;
  std::optional<Completed> settled;
};
WindowFrame render_to_window(Submission& submission, const Presenter& presenter,
                             Renderer& renderer,
                             const frame::FrameInputs& inputs,
                             CA::MetalDrawable* drawable);
std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs);
WindowFrame render_offscreen_frame(Submission& submission, Offscreen& target,
                                   Renderer& renderer,
                                   const frame::FrameInputs& inputs);

}  // namespace serenity::metal
