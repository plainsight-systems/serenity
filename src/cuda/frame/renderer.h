#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "core/animation/animate.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "cuda/acceleration/scene_acceleration.h"
#include "cuda/device/device.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/optix_context.h"
#include "cuda/device/submission.h"
#include "cuda/film/failure_count.h"
#include "cuda/frame/accumulation.h"
#include "cuda/frame/frame_images.h"
#include "cuda/passes/path/path.h"
#include "cuda/passes/test_pattern/test_pattern.h"
#include "cuda/passes/tone_map/tone_map.h"
#include "cuda/scene/light_glows.h"
#include "cuda/scene/scene_buffers.h"
#include "cuda/scene/shape_transforms.h"

namespace serenity::cuda {

// Axis: Frame graph.
//
// Carries out the core's schedule (core/frame/schedule.h) on CUDA: each of
// its passes, in its order, launched on the frame's stream, which orders
// them (no barrier between passes, unlike Metal 4's encoder). It decides
// nothing about what a frame computes; that is the schedule's and the
// core's (logical-overview.md, principle 10): the frame's constants
// (core/frame/frame_constants.h), its refusals (core/frame/refusals.h),
// which frames the accumulated image holds (core/frame/history.h) and where
// the frame's shapes are (core/animation/animate.h) are the core's, the
// same as Metal's (metal/frame/renderer.h).
//
// The backend is built a phase at a time (docs/research/2026-10-19-cuda-
// backend.md), and passes it does not build yet are refused, each by name,
// when the renderer is made, before any frame (the no-facade rule: a graph
// it cannot render never renders a part of itself). Built today: the test
// pattern, the path pass and the tone map, so the reference graph
// (graphs/path.toml).
//
// What it makes, by what the schedule has:
//
//   a pass that reads a scene   OptiX's context; the scene's arrays and
//                               block, its transforms and glows, its two
//                               structures (cuda/scene/, cuda/acceleration/)
//   a pass that accumulates     the accumulated image, by the core's rule,
//                               and the count of samples not finite
//                               (cuda/frame/accumulation.h, cuda/film/)
//   the tone map                the bloom pyramid (cuda/frame/frame_images.h)
//
// A frame: prepare() before its submission begins, so remaking an image
// for a new size may finish the frames in flight; then record(), which
// places and lights the frame's shapes in its slot's copies, uploads them
// and rebuilds what moves (animation::animate; scene_acceleration.h), then
// launches each pass. Nothing is allocated a frame (MEM.9).
//
// Throws CudaError, with the core's words for its refusals.
class Renderer {
 public:
  // Builds a pass for each entry of `schedule`. `scene` may be null when no
  // pass reads it. Throws CudaError if the schedule is invalid
  // (frame::invalid), a pass reads a scene and there is none, the scene
  // cannot be put on the GPU (frame::glower_refusal, the structures'), or
  // it has a pass this backend does not build yet.
  Renderer(const Device& device, Submission& submission,
           const frame::Schedule& schedule,
           const scene::SceneDescription* scene);

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;
  // Waits for the GPU before any pass's module or memory is released
  // (submission.h, Lifetime).
  ~Renderer();

  // Readies frame `inputs` at `size`: the core's refusals of its time and
  // camera, the accumulated image's history, the images at the size.
  // Called before the frame's submission begins; throws CudaError for a
  // refusal.
  void prepare(const frame::FrameInputs& inputs, frame::Extent size);

  // Records frame `inputs`, prepared at `target`'s size, on `begun`'s
  // stream, writing `target`. Throws CudaError if it was not prepared, or a
  // launch fails.
  void record(const FrameSlot& begun, const frame::FrameInputs& inputs,
              const Offscreen& target);

  // Samples a converging pass left out for not being finite, over every
  // completed frame (cuda/film/failure_count.h); 0 for a graph without one.
  std::uint64_t non_finite_samples() const noexcept;

  // The accumulated image, after every committed frame, into `out`, four
  // floats a pixel (cuda/device/images.h, PitchedImage::read). Throws
  // CudaError if the graph has none, or none has been rendered yet.
  void read_accumulated(std::span<float> out) const;

 private:
  // What prepare() readied for the frame, taken by record().
  struct Prepared {
    std::uint64_t index = 0;
    frame::Extent size;
    std::uint32_t accumulated_frames = 0;
  };

  void make_scene(const Device& device, const scene::SceneDescription& scene);
  // The frame's shapes placed and lights lit at its time, uploaded, and
  // the structures rebuilt where shapes move.
  void animate(const FrameSlot& begun, const frame::FrameInputs& inputs);
  PathLaunch path_launch(const FrameSlot& begun,
                         const frame::FrameInputs& inputs,
                         const Prepared& prepared, frame::Extent size) const;

  Submission& submission_;
  frame::Schedule schedule_;
  bool needs_scene_;
  std::optional<Prepared> prepared_;

  // The scene's, for a graph that reads it (released after the passes
  // that read it, and before OptiX's context).
  std::unique_ptr<OptixContext> optix_;
  std::unique_ptr<SceneBuffers> scene_;
  animation::Animation animation_;
  std::unique_ptr<ShapeTransforms> transforms_;
  std::unique_ptr<LightGlows> glows_;
  std::unique_ptr<SceneAcceleration> acceleration_;

  // The images and counts the passes share.
  std::unique_ptr<Accumulation> accumulation_;
  std::unique_ptr<FailureCount> non_finite_;
  std::unique_ptr<FrameImages> images_;

  // The passes.
  std::optional<TestPatternPass> test_pattern_;
  std::optional<PathPass> path_;
  std::optional<ToneMapPass> tone_map_;
};

// The first pass of `schedule` this backend does not build yet, refused by
// name: the reason, or none when it builds every one. A pure function of
// the schedule (F.8), so which passes the backend builds is a unit, tested
// as one.
std::optional<std::string> unbuilt_pass(const frame::Schedule& schedule);

// Readies, records and commits frame `inputs` into `target`, without
// waiting; returns its sequence, for Submission::wait_until_complete().
std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs);

}  // namespace serenity::cuda
