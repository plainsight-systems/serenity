#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "core/animation/animate.h"
#include "core/contracts/camera.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "cuda/acceleration/scene_acceleration.h"
#include "cuda/device/device.h"
#include "cuda/device/frame_upload.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/optix_context.h"
#include "cuda/device/submission.h"
#include "cuda/film/failure_count.h"
#include "cuda/frame/accumulation.h"
#include "cuda/frame/checkerboard_images.h"
#include "cuda/frame/feature_images.h"
#include "cuda/frame/frame_images.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/frame/mnee_list.h"
#include "cuda/frame/path_reuse_images.h"
#include "cuda/passes/checkerboard_fill/checkerboard_fill.h"
#include "cuda/passes/features/features.h"
#include "cuda/passes/path/path.h"
#include "cuda/passes/restir_pt_duplication/restir_pt_duplication.h"
#include "cuda/passes/restir_pt_initial/restir_pt_initial.h"
#include "cuda/passes/restir_pt_shade/restir_pt_shade.h"
#include "cuda/passes/restir_pt_spatial/restir_pt_spatial.h"
#include "cuda/passes/restir_pt_temporal/restir_pt_temporal.h"
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
// (graphs/path.toml); ReSTIR PT's five passes with MNEE, the features and
// the checkerboard's fill, so the data set's input graph
// (graphs/restir_pt_features.toml), and the features alone.
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
//   ReSTIR PT's passes          their arrays, settings and reuse textures,
//                               MNEE's work list and lost count, and with
//                               temporal reuse last frame's transforms and
//                               glows (cuda/frame/path_reuse_images.h,
//                               mnee_list.h)
//   the features pass           its images, and the frame before's
//                               transforms where shapes move
//                               (cuda/frame/feature_images.h)
//   a graph that feeds the      the fill and its history, ReSTIR PT's
//   network                     settings with checkerboard 1 (contract 20)
//                               (cuda/frame/checkerboard_images.h)
//
// A frame: prepare() before its submission begins, so remaking an image
// for a new size may finish the frames in flight; then record(), which
// places and lights the frame's shapes in its slot's copies, uploads them
// and rebuilds what moves (animation::animate; scene_acceleration.h),
// writes and uploads the frame's launch parameters once (frame_launch.h),
// then launches each pass in the schedule's order, the fill before the
// presenting pass in a graph that feeds the network, as Metal's. Nothing
// is allocated a frame (MEM.9).
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
  // stream, writing `target`, through a pinhole in a graph that feeds the
  // network (frame::traced_inputs). Throws CudaError if it was not prepared, or
  // a launch fails.
  void record(const FrameSlot& begun, const frame::FrameInputs& inputs,
              const Offscreen& target);

  // Samples a converging pass left out for not being finite, over every
  // completed frame (cuda/film/failure_count.h); 0 for a graph without one.
  std::uint64_t non_finite_samples() const noexcept;

  // MNEE requests the work list had no room for, their light lost
  // (contract 16), over every completed frame; 0 for a graph without
  // ReSTIR PT's passes.
  std::uint64_t mnee_lost_requests() const noexcept;

  // The image the graph shows, after every committed frame, into `out`,
  // four floats a pixel (cuda/device/images.h, PitchedImage::read): the
  // accumulated image, or in a graph that feeds the network the fill's
  // light, which it shows and the headless renderer writes, as Metal's.
  // Throws CudaError if the graph has none, or none has been rendered yet.
  void read_accumulated(std::span<float> out) const;

  // The feature images as the last frame wrote them, read back
  // (cuda/frame/feature_images.h). Throws CudaError if the graph has no
  // features pass or no frame has made its images yet.
  FeatureImages::Read read_features() const;

 private:
  // What prepare() readied for the frame, taken by record().
  struct Prepared {
    std::uint64_t index = 0;
    frame::Extent size;
    std::uint32_t accumulated_frames = 0;
    // the reuse history's frames before this one
    std::uint64_t reused_frames = 0;
    // the checkerboard's fill reads the frame before's light
    bool fill_history = false;
  };

  // Each family's state, one struct each, present only for a graph with
  // the family's passes (C.1, R.5), as Metal's (metal/frame/renderer.h).
  //
  // ReSTIR PT's: its arrays, its settings and the reuse textures.
  struct RestirPtFamily {
    RestirPtFamily(Submission& submission,
                   const contracts::PathReuseSettings& settings,
                   std::span<const std::uint16_t> textures, bool spatial,
                   bool temporal, std::uint32_t lights);

    PathReuseImages images;
    contracts::PathReuseSettings settings;
    DeviceBuffer pairing;  // the reuse textures, one zero texel without
  };
  // MNEE's (contract 16): its work list and its count of lost requests.
  struct MneeFamily {
    explicit MneeFamily(Submission& submission);

    MneeList list;
    FailureCount lost;
  };
  // The features pass's: its images and the shapes as the frame before
  // placed them, when shapes move.
  struct FeatureFamily {
    FeatureFamily(Submission& submission, const scene::SceneDescription& scene,
                  bool moves);

    FeatureImages images;
    std::unique_ptr<ShapeTransforms> previous_transforms;
  };

  void make_scene(const Device& device, const scene::SceneDescription& scene);
  // ReSTIR PT's and MNEE's, for a schedule with ReSTIR PT's settings,
  // `settings`.
  void make_restir_pt(const frame::Schedule& schedule,
                      const contracts::PathReuseSettings& settings,
                      const scene::SceneDescription& scene);
  // The frame's shapes placed and lights lit at its time, uploaded, and
  // the structures rebuilt where shapes move; with them last frame's and
  // the features' frame before's where the graph keeps them.
  void animate(const FrameSlot& begun, const frame::FrameInputs& inputs,
               bool previous);
  FrameLaunch frame_launch(const FrameSlot& begun,
                           const frame::FrameInputs& inputs,
                           const Prepared& prepared, frame::Extent size,
                           bool previous) const;
  // The pass at `kind`, recorded on `begun`'s stream.
  void record_pass(frame::PassKind kind, const FrameSlot& begun,
                   const FrameLaunch& launch, const frame::FrameInputs& inputs,
                   const Prepared& prepared, const Offscreen& target);
  // The image the presenting pass shows and read_accumulated() reads: the
  // fill's in a graph that feeds the network, else the accumulated image;
  // null before the first frame.
  const PitchedImage* shown() const noexcept;

  Submission& submission_;
  frame::Schedule schedule_;
  // The frame's launch parameters, staged a slot and uploaded once a frame
  // (frame_launch.h).
  FrameUpload launch_;
  bool needs_scene_;
  // ReSTIR PT's temporal reuse merges last frame (frame::ReuseHistory)
  bool reuses_history_ = false;
  // the graph feeds the network: checkerboarded, ending in the fill
  bool feeds_network_ = false;
  std::optional<Prepared> prepared_;
  // What the next frame's temporal merge asks of this one.
  std::optional<contracts::Camera> previous_camera_;
  frame::Seconds previous_time_{0.0};

  // The scene's, for a graph that reads it (released after the passes
  // that read it, and before OptiX's context).
  std::unique_ptr<OptixContext> optix_;
  std::unique_ptr<SceneBuffers> scene_;
  animation::Animation animation_;
  std::unique_ptr<ShapeTransforms> transforms_;
  std::unique_ptr<LightGlows> glows_;
  // Last frame's, for a temporal graph whose scene changes.
  std::unique_ptr<ShapeTransforms> last_transforms_;
  std::unique_ptr<LightGlows> last_glows_;
  std::unique_ptr<SceneAcceleration> acceleration_;

  // The images and counts the passes share.
  std::unique_ptr<Accumulation> accumulation_;
  std::unique_ptr<FailureCount> non_finite_;
  std::unique_ptr<FrameImages> images_;
  std::optional<RestirPtFamily> restir_pt_;
  std::optional<MneeFamily> mnee_;
  std::optional<FeatureFamily> features_;
  std::unique_ptr<CheckerboardImages> checkerboard_;

  // The passes.
  std::optional<TestPatternPass> test_pattern_;
  std::optional<PathPass> path_;
  std::optional<ToneMapPass> tone_map_;
  std::optional<RestirPtInitialPass> restir_pt_initial_;
  std::optional<RestirPtTemporalPass> restir_pt_temporal_;
  std::optional<RestirPtSpatialPass> restir_pt_spatial_;
  std::optional<RestirPtShadePass> restir_pt_shade_;
  std::optional<RestirPtDuplicationPass> restir_pt_duplication_;
  std::optional<FeaturesPass> features_pass_;
  std::optional<CheckerboardFillPass> fill_;
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
