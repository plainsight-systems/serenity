#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/animation/animate.h"
#include "core/contracts/camera.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/denoise/network.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/sampling/jitter.h"
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
#include "cuda/frame/denoise_images.h"
#include "cuda/frame/depth_of_field_images.h"
#include "cuda/frame/feature_images.h"
#include "cuda/frame/frame_images.h"
#include "cuda/frame/frame_launch.h"
#include "cuda/frame/mnee_list.h"
#include "cuda/frame/path_reuse_images.h"
#include "cuda/passes/checkerboard_fill/checkerboard_fill.h"
#include "cuda/passes/denoise/denoise.h"
#include "cuda/passes/depth_of_field/depth_of_field.h"
#include "cuda/passes/features/features.h"
#include "cuda/passes/path/path.h"
#include "cuda/passes/restir_pt_duplication/restir_pt_duplication.h"
#include "cuda/passes/restir_pt_initial/restir_pt_initial.h"
#include "cuda/passes/restir_pt_shade/restir_pt_shade.h"
#include "cuda/passes/restir_pt_spatial/restir_pt_spatial.h"
#include "cuda/passes/restir_pt_temporal/restir_pt_temporal.h"
#include "cuda/passes/test_pattern/test_pattern.h"
#include "cuda/passes/tone_map/tone_map.h"
#include "cuda/present/presenter.h"
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
// backend.md), and passes it does not build are refused, each by name,
// when the renderer is made, before any frame (the no-facade rule: a graph
// it cannot render never renders a part of itself). Built: every pass the
// core schedules: the test pattern, the path pass and the tone map, so the
// reference graph (graphs/path.toml); ReSTIR PT's five passes with MNEE,
// the features and the checkerboard's fill, so the data set's input graph
// (graphs/restir_pt_features.toml) and the features alone; and the
// denoiser and the depth of field, so the demo graph
// (graphs/restir_pt_denoised.toml).
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
//   the denoise pass            the network, read once (core/denoise/
//                               network.h); its images and the slots'
//                               hand-offs (cuda/frame/denoise_images.h); a
//                               set of features a slot at the light size
//                               and one at the output's, the full-size
//                               albedo; the radiance image it writes at the
//                               output's size
//   the depth of field          its images (cuda/frame/
//                               depth_of_field_images.h)
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
// A denoised frame (Metal's metal/frame/renderer.h, "The denoise pass" and
// "Two halves"; docs/research/2026-10-19-cuda-backend.md, "Phase 5's
// design"). It renders at two sizes (frame::light_size): every pass before
// the denoise pass at the light size, half the output's each way, and the
// denoise pass and those after it at the output's. Two launch structs a
// frame: the light's, which the light half's passes read, and the
// output's, framed at the output's size, its feature images the slot's
// full-size set. The frame splits at the denoise pass: the light half,
// every pass before it, then the full-size albedo (the features pass
// launched again with the output's struct), then the fill, which writes
// the slot's hand-off of the light; and the image half, the denoise pass,
// the depth of field and the tone map. parts() is Parts::two for such a
// graph and Parts::one for any other; in a frame begun in two parts,
// record() hands the frame off between its halves (submission.h, "Two
// parts") and records the image half on the image stream, so frame N's
// image half runs beside frame N + 1's light half; in one part, both
// halves on the light stream, the same work in the same order, so the
// same pixels. The hand-off, what the image half reads of the light
// half's, is ringed by the slot, as Metal's: the light-size features, the
// full-size features and the light's radiance, one each a slot (CP.3:
// frame N + 1's light half never writes what frame N's image half reads;
// GPU.7). What the image half writes (the denoiser's history, the
// radiance image, the bloom, the depth of field's images) the light half
// never reads, and each image half follows the one before on its stream.
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
  // network (frame::traced_inputs); a denoised frame begun in two parts
  // handed off between its halves (above). Throws CudaError if it was not
  // prepared, it was begun in two parts for a graph of one, or a launch
  // fails.
  void record(const FrameSlot& begun, const frame::FrameInputs& inputs,
              RenderTarget target);

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

  // The parts a frame of this graph is begun in (above).
  Parts parts() const noexcept;

  // The network's light input as the last frame recorded it, after every
  // committed frame, and the jitter it was traced through, as Metal's:
  // the fill's light in a graph that feeds the network, which a denoised
  // graph hands its network and the data set's input graph writes, so a
  // test holds the two to the bit. RGBA32Float a pixel, rows from the top.
  // Throws CudaError if the graph's light is not the network's input, or
  // no frame has been recorded.
  struct NetworkInput {
    frame::Extent size;
    std::vector<float> rgba;
    sampling::CameraJitter jitter;
  };
  NetworkInput read_network_input() const;

 private:
  // What prepare() readied for the frame, taken by record().
  struct Prepared {
    std::uint64_t index = 0;
    frame::Extent size;   // the output's
    frame::Extent light;  // the light half's: the output's but when denoised
    std::uint32_t accumulated_frames = 0;
    // the reuse history's frames before this one
    std::uint64_t reused_frames = 0;
    // the checkerboard's fill reads the frame before's light
    bool fill_history = false;
    // the denoiser reads last frame's output
    bool denoise_history = false;
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
  // The features pass's: its images, one set, or one a slot when the
  // denoiser reads them from the image half (above); the slot the last
  // frame wrote; and the shapes as the frame before placed them, when
  // shapes move.
  struct FeatureFamily {
    enum class Sets : std::uint8_t { one, per_slot };
    FeatureFamily(Submission& submission, const scene::SceneDescription& scene,
                  bool moves, Sets sets);

    // The set frame slot `slot` writes.
    FeatureImages& images(std::uint32_t slot);
    const FeatureImages& images(std::uint32_t slot) const;

    Sets kept;
    std::array<std::unique_ptr<FeatureImages>, frames_in_flight> sets;
    std::uint32_t last_slot = 0;
    std::unique_ptr<ShapeTransforms> previous_transforms;
  };
  // The denoise pass's: the network's last width, its pass and images, a
  // set of full-size features a slot, and the radiance image it writes at
  // the output's size, which the depth of field and the tone map read.
  struct DenoiseFamily {
    DenoiseFamily(const Device& device, Submission& submission,
                  const denoise::Network& network);

    std::uint32_t features;
    DenoisePass pass;
    DenoiseImages images;
    std::array<std::unique_ptr<FeatureImages>, frames_in_flight> full_features;
    std::unique_ptr<PitchedImage> radiance;
  };
  // The depth of field's: its pass and images.
  struct DepthOfFieldFamily {
    explicit DepthOfFieldFamily(Submission& submission);

    DepthOfFieldPass pass;
    DepthOfFieldImages images;
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
  // A denoised frame's output struct (above): `light`'s, framed at the
  // output's `size`, its features the slot's full-size set.
  FrameLaunch output_launch(const FrameSlot& begun,
                            const frame::FrameInputs& inputs,
                            const FrameLaunch& light, frame::Extent size) const;
  // A denoised frame's image half on `stream`: the denoise pass, the depth
  // of field through the frame's own `lens` and the tone map, into
  // `target`.
  void record_image(CUstream stream, const FrameSlot& begun,
                    const frame::FrameInputs& inputs,
                    const contracts::Camera& lens, const Prepared& prepared,
                    const FrameLaunch& light, const FrameLaunch& output,
                    RenderTarget target);
  // The denoise pass, reading the light's struct's features and the
  // output's full-size albedo; the depth of field through `lens`.
  void record_denoise(CUstream stream, const FrameSlot& begun,
                      const frame::FrameInputs& inputs,
                      const Prepared& prepared, const FrameLaunch& light,
                      const FrameLaunch& output) const;
  void record_depth_of_field(CUstream stream, const FrameSlot& begun,
                             const contracts::Camera& lens,
                             const Prepared& prepared) const;
  // The pass at `kind`, recorded on `stream`, one of `begun`'s; every kind
  // but the denoise pass and the depth of field, which record_image()
  // records.
  void record_pass(frame::PassKind kind, CUstream stream,
                   const FrameSlot& begun, const FrameLaunch& launch,
                   const frame::FrameInputs& inputs, const Prepared& prepared,
                   RenderTarget target);
  // The image the presenting pass shows and read_accumulated() reads: the
  // fill's in a graph that feeds the network, else the accumulated image;
  // null before the first frame.
  const PitchedImage* shown() const noexcept;

  const Device& device_;
  Submission& submission_;
  frame::Schedule schedule_;
  // The frame's launch parameters, staged a slot and uploaded once a frame
  // (frame_launch.h); a denoised graph's output struct beside them.
  FrameUpload launch_;
  std::unique_ptr<FrameUpload> output_launch_;
  bool needs_scene_;
  // ReSTIR PT's temporal reuse merges last frame (frame::ReuseHistory)
  bool reuses_history_ = false;
  // the graph feeds the network: checkerboarded, ending in the fill
  bool feeds_network_ = false;
  std::optional<Prepared> prepared_;
  // What the next frame's temporal merge asks of this one.
  std::optional<contracts::Camera> previous_camera_;
  frame::Seconds previous_time_{0.0};
  // What read_network_input() reads of the last frame recorded.
  std::optional<std::uint32_t> last_slot_;
  sampling::CameraJitter last_jitter_{};

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
  std::unique_ptr<DenoiseFamily> denoise_;
  std::unique_ptr<DepthOfFieldFamily> depth_of_field_;

  // The passes.
  std::optional<TestPatternPass> test_pattern_;
  std::optional<PathPass> path_;
  std::optional<ToneMapPass> tone_map_;
  std::optional<RestirPtInitialPass> restir_pt_initial_;
  std::optional<RestirPtTemporalPass> restir_pt_temporal_;
  std::optional<RestirPtSpatialPass> restir_pt_spatial_;
  // only without the spatial pass, whose merge shades
  std::optional<RestirPtShadePass> restir_pt_shade_;
  std::optional<RestirPtDuplicationPass> restir_pt_duplication_;
  std::optional<FeaturesPass> features_pass_;
  std::optional<CheckerboardFillPass> fill_;
};

// Readies, records and commits frame `inputs` into `target`, begun in the
// renderer's parts, without waiting; returns its sequence, for
// Submission::wait_until_complete().
std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs);

// Readies, records and commits frame `inputs` into `image`, the
// swapchain image the presenter acquired for the frame's slot, and
// presents it (cuda/present/presenter.h): CUDA's wait for the slot's last
// copy on the stream that writes the target, the frame, and the signal and
// present after its last work; returns its sequence. Throws CudaError if
// the frame does not take `image`'s slot, so the slot's buffer is never
// written by another frame than it was acquired for (I.6).
std::uint64_t render_to_window(Submission& submission, Presenter& presenter,
                               Renderer& renderer,
                               const frame::FrameInputs& inputs,
                               const WindowImage& image);

}  // namespace serenity::cuda
