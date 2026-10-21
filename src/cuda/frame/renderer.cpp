#include "cuda/frame/renderer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "core/camera/thin_lens.h"
#include "core/contracts/linear_image.h"
#include "core/denoise/network.h"
#include "core/frame/frame_constants.h"
#include "core/frame/refusals.h"
#include "core/passes/depth_of_field.h"
#include "core/sampling/pairing.h"
#include "cuda/acceleration/mesh_levels.h"
#include "cuda/device/error.h"

namespace serenity::cuda {

namespace {

// A refusal the core gives, thrown as this backend's error: `prefix` and
// the reason. None, nothing.
void refuse(const std::optional<std::string>& reason,
            std::string_view prefix = "Renderer: ") {
  if (reason) {
    throw CudaError(std::string(prefix) + *reason);
  }
}

// The pass `pass` the renderer made for a kind of its schedule, or
// CudaError: a pass is made for every kind the schedule has (the
// constructor), so this never throws for one it records.
template <typename Pass>
Pass& made(std::optional<Pass>& pass, std::string_view name) {
  if (!pass) {
    throw CudaError("Renderer: no " + std::string(name) + " pass was made");
  }
  return *pass;
}

// The launch parameters every slot's copy starts as, before its first
// frame: zeroes, unread.
std::span<const std::byte> no_launch() {
  static constexpr FrameLaunch none{};
  return std::as_bytes(std::span(&none, 1));
}

// Whether `schedule` has a pass of `kind`.
bool has(const frame::Schedule& schedule, frame::PassKind kind) {
  return std::ranges::find(schedule.passes, kind) != schedule.passes.end();
}

// A device address of `T`s in the launch parameters, from a device
// address (frame_launch.h: on the host, every field is one).
constexpr std::uint64_t at(CUdeviceptr address) noexcept { return address; }

}  // namespace

Renderer::RestirPtFamily::RestirPtFamily(
    Submission& submission, const contracts::PathReuseSettings& settings,
    std::span<const std::uint16_t> textures, bool spatial, bool temporal,
    std::uint32_t lights)
    : images(submission,
             spatial ? PathReuseImages::Spatial::merged
                     : PathReuseImages::Spatial::none,
             temporal ? PathReuseImages::Temporal::merged
                      : PathReuseImages::Temporal::none,
             lights),
      settings(settings),
      pairing(textures.size_bytes()) {
  upload_now(pairing.address(), std::as_bytes(textures), "Renderer");
}

Renderer::MneeFamily::MneeFamily(Submission& submission)
    : list(submission),
      lost(submission, "MNEE requests lost to a full work list") {}

Renderer::FeatureFamily::FeatureFamily(Submission& submission,
                                       const scene::SceneDescription& scene,
                                       bool moves, Sets sets)
    : kept(sets) {
  const std::size_t made =
      sets == Sets::per_slot ? std::size_t{frames_in_flight} : 1u;
  for (std::size_t s = 0; s < made; ++s) {
    this->sets.at(s) = std::make_unique<FeatureImages>(submission);
  }
  if (moves) {
    previous_transforms =
        std::make_unique<ShapeTransforms>(submission, scene.shapes.transforms);
  }
}

FeatureImages& Renderer::FeatureFamily::images(std::uint32_t slot) {
  return *sets.at(kept == Sets::per_slot ? slot : 0u);
}

const FeatureImages& Renderer::FeatureFamily::images(std::uint32_t slot) const {
  return *sets.at(kept == Sets::per_slot ? slot : 0u);
}

// The denoise pass's (renderer.h): the network's last width, its pass and
// images, and the full-size albedo's set a slot.
Renderer::DenoiseFamily::DenoiseFamily(const Device& device,
                                       Submission& submission,
                                       const denoise::Network& network)
    : features(network.widths.dec.back()),
      pass(network),
      images(device, submission, network.widths) {
  for (std::unique_ptr<FeatureImages>& set : full_features) {
    set = std::make_unique<FeatureImages>(submission);
  }
}

Renderer::DepthOfFieldFamily::DepthOfFieldFamily(Submission& submission)
    : images(submission) {}

Renderer::Renderer(const Device& device, Submission& submission,
                   const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : device_(device),
      submission_(submission),
      schedule_(schedule),
      launch_(submission, no_launch()),
      output_launch_(frame::denoises(schedule) ? std::make_unique<FrameUpload>(
                                                     submission, no_launch())
                                               : nullptr),
      needs_scene_(std::ranges::any_of(schedule.passes, frame::needs_scene)),
      feeds_network_(frame::feeds_network(schedule)) {
  // The core decides which schedules can be carried out (principle 10).
  refuse(frame::invalid(schedule));
  refuse(frame::scene_refusal(schedule, scene));
  if (schedule.restir_pt) {
    make_restir_pt(schedule, *schedule.restir_pt, *scene);
  }
  if (needs_scene_) {
    make_scene(device, *scene);
  }
  if (frame::accumulates(schedule)) {
    accumulation_ = std::make_unique<Accumulation>(device, submission_);
    non_finite_ =
        std::make_unique<FailureCount>(submission_, "samples not finite");
  }
  if (schedule.tone_map) {
    images_ = std::make_unique<FrameImages>(submission_);
  }
  const bool denoises = frame::denoises(schedule);
  if (frame::writes_features(schedule)) {
    // A set a slot when the denoiser reads them from the image half
    // (renderer.h, "A denoised frame").
    features_.emplace(
        submission_, *scene, animation::moves(animation_),
        denoises ? FeatureFamily::Sets::per_slot : FeatureFamily::Sets::one);
  }
  if (feeds_network_) {
    // A denoised graph's fill writes the slot's hand-off, not its own image.
    checkerboard_ = std::make_unique<CheckerboardImages>(
        device, submission_,
        denoises ? CheckerboardImages::Filled::none
                 : CheckerboardImages::Filled::own);
    fill_.emplace();
  }
  // The network, read once (core/denoise/network.h); frame::invalid has
  // made sure a denoised graph has its settings, the features pass and the
  // depth of field.
  if (denoises) {
    if (!schedule.denoise) {
      throw CudaError("Renderer: a denoise pass with no settings");
    }
    denoise_ = std::make_unique<DenoiseFamily>(
        device, submission_, denoise::load_network(schedule.denoise->weights));
  }
  if (has(schedule, frame::PassKind::depth_of_field)) {
    depth_of_field_ = std::make_unique<DepthOfFieldFamily>(submission_);
  }
  const std::array<Pipeline::HitRecord, structure_records> records =
      acceleration_
          ? acceleration_->hit_records(scene_->block(), transforms_->address())
          : std::array<Pipeline::HitRecord, structure_records>{};
  for (const frame::PassKind kind : schedule.passes) {
    switch (kind) {
      case frame::PassKind::test_pattern:
        test_pattern_.emplace();
        break;
      case frame::PassKind::path:
        path_.emplace(*optix_, records);
        break;
      case frame::PassKind::tone_map:
        // frame::invalid refuses a tone map without its settings.
        if (!schedule.tone_map) {
          throw CudaError("Renderer: a tone map with no settings");
        }
        tone_map_.emplace(*schedule.tone_map);
        break;
      case frame::PassKind::restir_pt_initial:
        restir_pt_initial_.emplace(*optix_, records);
        break;
      case frame::PassKind::restir_pt_temporal:
        restir_pt_temporal_.emplace(*optix_, records);
        break;
      case frame::PassKind::restir_pt_spatial:
        restir_pt_spatial_.emplace(*optix_, records);
        break;
      case frame::PassKind::restir_pt_shade:
        restir_pt_shade_.emplace();
        break;
      case frame::PassKind::restir_pt_duplication:
        restir_pt_duplication_.emplace();
        break;
      case frame::PassKind::features:
        features_pass_.emplace(*optix_, records);
        break;
      case frame::PassKind::denoise:
      case frame::PassKind::depth_of_field:
        break;  // their families, above
    }
  }
}

// ReSTIR PT's and MNEE's, only for their graph, as Metal's
// (metal/frame/renderer.cpp, make_restir_pt, make_mnee).
void Renderer::make_restir_pt(const frame::Schedule& schedule,
                              const contracts::PathReuseSettings& reuse,
                              const scene::SceneDescription& scene) {
  // What the G-buffer, the light tiles and MNEE's requests pack, refused
  // rather than wrapped (I.5).
  refuse(frame::restir_pt_refusal(scene));
  refuse(frame::mnee_refusal(scene));
  const bool temporal = has(schedule, frame::PassKind::restir_pt_temporal);
  reuses_history_ = temporal;
  const bool spatial = has(schedule, frame::PassKind::restir_pt_spatial);
  contracts::PathReuseSettings settings = reuse;
  // the network's input is traced checkerboarded (contract 20)
  settings.checkerboard = feeds_network_ ? 1u : 0u;
  // With spatial reuse, the three reuse textures, built once on the CPU
  // from the graph's sigma and the core's seed (core/sampling/pairing.h);
  // one zero texel without.
  const double sigma =
      sampling::reuse_texture_sigma(settings.sigma, feeds_network_);
  const std::vector<std::uint16_t> textures =
      spatial ? sampling::reuse_textures(sigma, sampling::reuse_texture_seed)
              : std::vector<std::uint16_t>(1, 0u);
  restir_pt_.emplace(submission_, settings, textures, spatial, temporal,
                     scene.light_counts.lights);
  mnee_.emplace(submission_);
}

// The scene on the GPU, for a graph that reads it, and what its frames
// place, as Metal's (metal/frame/renderer.cpp, make_scene).
void Renderer::make_scene(const Device& device,
                          const scene::SceneDescription& scene) {
  optix_ = std::make_unique<OptixContext>(device);
  scene_ = std::make_unique<SceneBuffers>(submission_, scene);
  animation_ = scene.animation;
  transforms_ =
      std::make_unique<ShapeTransforms>(submission_, scene.shapes.transforms);
  // Checked here, not by animate() in a frame, so nothing about the scene
  // can fail once a frame's submission has begun.
  refuse(frame::glower_refusal(scene));
  glows_ =
      std::make_unique<LightGlows>(submission_, scene.light_counts.spheres);
  // Last frame's, for a temporal graph whose scene changes.
  if (reuses_history_ && animation::changes(animation_)) {
    last_transforms_ =
        std::make_unique<ShapeTransforms>(submission_, scene.shapes.transforms);
    last_glows_ =
        std::make_unique<LightGlows>(submission_, scene.light_counts.spheres);
  }
  // Where the families meet: the structures learn which shapes move, by
  // the movers' targets, and which are lights, by the scene's answer for
  // each shape (contract 3), and nothing else of either: the lights are
  // the shapes a ray carrying the occluders skips (contract 19).
  std::vector<std::uint32_t> moving;
  moving.reserve(animation_.movers.size());
  for (const animation::Mover& mover : animation_.movers) {
    moving.push_back(mover.target);
  }
  const std::vector<std::uint32_t> lights = scene::light_shapes(scene);
  // Each sphere's mesh, by its error (mesh_levels.h).
  const std::vector<std::uint8_t> levels = sphere_levels(scene);
  acceleration_ = std::make_unique<SceneAcceleration>(
      *optix_, submission_, scene.shapes,
      SceneAcceleration::Roles{
          .moving = moving, .lights = lights, .levels = levels});
}

Renderer::~Renderer() { submission_.wait_idle(); }

void Renderer::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
  refuse(frame::time_refusal(inputs.time, inputs.index, "its time"), "");
  refuse(frame::camera_refusal(inputs, needs_scene_));
  // The frame before, which the features' motion is measured from: its
  // camera framed and its time placing the shapes in record(), so both are
  // checked here, before the submission begins.
  if (features_ && inputs.previous) {
    refuse(frame::time_refusal(inputs.previous->time, inputs.index,
                               "the frame before's time"),
           "");
    refuse(frame::previous_camera_refusal(inputs));
  }
  prepared_.reset();
  // Every image before the denoise pass at the light size; the output's
  // own in a graph without it (frame::light_size).
  const frame::Extent light = frame::light_size(schedule_, size);
  if (images_) {
    images_->prepare(size);
  }
  if (features_) {
    for (const std::unique_ptr<FeatureImages>& set : features_->sets) {
      if (set) {
        set->prepare(light);
      }
    }
  }
  const bool fill_history =
      checkerboard_ ? checkerboard_->prepare(inputs, light) : false;
  if (mnee_) {
    mnee_->list.prepare(light);
  }
  // 0 without temporal reuse.
  const std::uint64_t reused =
      restir_pt_ ? restir_pt_->images.prepare(inputs, light) : 0u;
  std::uint32_t held = 0;
  if (accumulation_) {
    const frame::SceneMotion motion = animation::changes(animation_)
                                          ? frame::SceneMotion::changing
                                          : frame::SceneMotion::still;
    held = accumulation_->prepare(inputs, light, motion);
  }
  bool denoise_history = false;
  if (denoise_) {
    denoise_history = denoise_->images.prepare(inputs, light);
    for (const std::unique_ptr<FeatureImages>& set : denoise_->full_features) {
      set->prepare(size);
    }
    if (!denoise_->radiance || !(denoise_->radiance->size() == size)) {
      if (denoise_->radiance) {
        submission_.finish();  // frames in flight may still read it
      }
      denoise_->radiance.reset();
      denoise_->radiance =
          std::make_unique<PitchedImage>(device_, submission_, size);
    }
  }
  if (depth_of_field_) {
    depth_of_field_->images.prepare(size);
  }
  prepared_ = Prepared{inputs.index, size,         light,          held,
                       reused,       fill_history, denoise_history};
}

void Renderer::animate(const FrameSlot& begun, const frame::FrameInputs& inputs,
                       bool previous) {
  if (!scene_ || !animation::changes(animation_)) {
    return;
  }
  const std::span<contracts::Transform> placed =
      transforms_->transforms(begun.slot);
  animation::animate(animation_, inputs.time, placed,
                     glows_->glows(begun.slot));
  transforms_->upload(begun);
  glows_->upload(begun);
  if (acceleration_->moves()) {
    acceleration_->update(begun, placed);
  }
  // Last frame's, at its time; unread at a history's first frame.
  if (last_transforms_ && previous) {
    animation::animate(animation_, previous_time_,
                       last_transforms_->transforms(begun.slot),
                       last_glows_->glows(begun.slot));
    last_transforms_->upload(begun);
    last_glows_->upload(begun);
  }
  // The features' frame before, at its time; unread without one. Its
  // shapes alone: a glow does not move a surface.
  if (features_ && features_->previous_transforms && inputs.previous) {
    animation::place(animation_, inputs.previous->time,
                     features_->previous_transforms->transforms(begun.slot));
    features_->previous_transforms->upload(begun);
  }
}

// The frame's launch parameters (frame_launch.h): its constants and camera
// framed for `size`, and the address of each of its arrays the graph has;
// 0 for the rest. `previous`: whether a temporal graph merges last frame.
FrameLaunch Renderer::frame_launch(const FrameSlot& begun,
                                   const frame::FrameInputs& inputs,
                                   const Prepared& prepared, frame::Extent size,
                                   bool previous) const {
  FrameLaunch launch{};
  launch.frame =
      frame::frame_constants(inputs, size, prepared.accumulated_frames,
                             frame::camera_jitter(schedule_, inputs.index));
  if (inputs.camera) {
    launch.camera = camera::shader_form(*inputs.camera, size);
  }
  if (scene_) {
    launch.structures = acceleration_->block();
    launch.scene = scene_->block_address();
    launch.transforms = transforms_->address();
    launch.glows = glows_->address();
  }
  if (accumulation_) {
    const PitchedImage* image = accumulation_->image();
    if (image == nullptr) {
      throw CudaError("Renderer: the frame has no accumulated image");
    }
    launch.accumulated = PitchedRows{image->address(), image->pitch()};
    launch.non_finite = non_finite_->address(begun.slot);
  }
  if (restir_pt_) {
    const PathReuseImages::Addresses arrays =
        restir_pt_->images.addresses(inputs.index);
    launch.path_settings = restir_pt_->settings;
    if (reuses_history_) {
      // Whether there is a frame before to merge, with its camera framed
      // for this frame's size.
      launch.path_constants = frame::path_reuse_constants(
          previous ? previous_camera_ : std::nullopt, size);
    }
    const frame::Extent traced = frame::traced_grid(size, feeds_network_);
    launch.traced = uint2{traced.width, traced.height};
    launch.pairing = at(restir_pt_->pairing.address());
    launch.path_reservoirs = at(arrays.reservoirs);
    launch.primaries = at(arrays.primaries);
    launch.light = at(arrays.light);
    launch.shifts = at(arrays.shifts);
    launch.replays = at(arrays.replays);
    launch.replay_counter = at(arrays.counter);
    launch.light_tiles = at(arrays.light_tiles);
    launch.light_power = at(arrays.light_power);
    launch.last_light_power = at(arrays.last_light_power);
    launch.last_primaries = at(arrays.last_primaries);
    launch.path_finals = at(arrays.finals);
    launch.last_path_finals = at(arrays.last_finals);
    launch.previous_positions = at(arrays.previous_positions);
    launch.previous_pixels = at(arrays.previous_pixels);
    launch.duplication = at(arrays.duplication);
    // Last frame's transforms and glows: this frame's own where the scene
    // is still, else animated at the previous time.
    if (reuses_history_) {
      launch.last_transforms =
          last_transforms_ ? last_transforms_->address() : launch.transforms;
      launch.last_glows = last_glows_ ? last_glows_->address() : launch.glows;
    }
  }
  if (mnee_) {
    launch.mnee_requests = at(mnee_->list.requests());
    launch.mnee_header = at(mnee_->list.header());
    launch.mnee_lost = mnee_->lost.address(begun.slot);
  }
  if (features_) {
    // Whether there is a frame before, and its camera framed for this
    // frame's size; its shapes, animated where they move.
    launch.feature_constants = frame::feature_constants(inputs.previous, size);
    launch.feature_last_transforms =
        features_->previous_transforms
            ? features_->previous_transforms->address()
            : launch.transforms;
    const FeatureImages& set = features_->images(begun.slot);
    launch.albedo = at(set.albedo());
    launch.normal = at(set.normal());
    launch.depth = at(set.depth());
    launch.motion = at(set.motion());
  }
  if (checkerboard_) {
    // The frame before's history is this history's frame before when the
    // fill's own history holds it.
    launch.checkerboard_constants =
        frame::checkerboard_constants(prepared.fill_history);
    const PitchedImage* before = checkerboard_->before(prepared.index);
    const PitchedImage* written = checkerboard_->written(prepared.index);
    // The fill's light: the slot's hand-off in a denoised graph
    // (renderer.h, "A denoised frame"), else its own image.
    const PitchedImage& filled =
        denoise_ ? denoise_->images.light_radiance(begun.slot)
                 : *checkerboard_->filled();
    launch.previous_light = PitchedRows{before->address(), before->pitch()};
    launch.history_light = PitchedRows{written->address(), written->pitch()};
    launch.filled = PitchedRows{filled.address(), filled.pitch()};
  }
  return launch;
}

FrameLaunch Renderer::output_launch(const FrameSlot& begun,
                                    const frame::FrameInputs& inputs,
                                    const FrameLaunch& light,
                                    frame::Extent size) const {
  FrameLaunch launch = light;
  // At the output's size, unjittered, as the training's full-size albedo
  // was rendered (Metal's denoise_resources()); the light's accumulated
  // count, which nothing at this size reads.
  launch.frame = frame::frame_constants(
      inputs, size, light.frame.accumulated_frames, std::nullopt);
  if (inputs.camera) {
    launch.camera = camera::shader_form(*inputs.camera, size);
  }
  launch.feature_constants = frame::feature_constants(inputs.previous, size);
  const FeatureImages& full = *denoise_->full_features.at(begun.slot);
  launch.albedo = at(full.albedo());
  launch.normal = at(full.normal());
  launch.depth = at(full.depth());
  launch.motion = at(full.motion());
  return launch;
}

const PitchedImage* Renderer::shown() const noexcept {
  if (denoise_) {
    return denoise_->radiance.get();
  }
  if (checkerboard_) {
    return checkerboard_->filled();
  }
  return accumulation_ ? accumulation_->image() : nullptr;
}

Parts Renderer::parts() const noexcept {
  return denoise_ ? Parts::two : Parts::one;
}

void Renderer::record_pass(frame::PassKind kind, CUstream stream,
                           const FrameSlot& begun, const FrameLaunch& launch,
                           const frame::FrameInputs& inputs,
                           const Prepared& prepared, RenderTarget target) {
  const CUdeviceptr address = launch_.address();
  switch (kind) {
    case frame::PassKind::test_pattern:
      made(test_pattern_, "test pattern")
          .record(
              stream,
              frame::frame_constants(inputs, prepared.size,
                                     prepared.accumulated_frames, std::nullopt),
              target.address);
      break;
    case frame::PassKind::path:
      made(path_, "path").record(stream, address, prepared.size);
      break;
    case frame::PassKind::tone_map: {
      const PitchedImage* radiance = shown();
      if (radiance == nullptr) {
        throw CudaError("Renderer: the tone map has no image to show");
      }
      made(tone_map_, "tone map")
          .record(stream, *radiance, *images_, target.address);
      break;
    }
    case frame::PassKind::restir_pt_initial:
      made(restir_pt_initial_, "ReSTIR PT's initial")
          .record(stream, address, launch);
      break;
    case frame::PassKind::restir_pt_temporal:
      made(restir_pt_temporal_, "ReSTIR PT's temporal")
          .record(stream, address, launch);
      break;
    case frame::PassKind::restir_pt_spatial:
      made(restir_pt_spatial_, "ReSTIR PT's spatial")
          .record(stream, address, launch);
      break;
    case frame::PassKind::restir_pt_shade:
      made(restir_pt_shade_, "ReSTIR PT's shade")
          .record(stream, address, launch);
      break;
    case frame::PassKind::restir_pt_duplication:
      made(restir_pt_duplication_, "ReSTIR PT's duplication")
          .record(stream, address, launch);
      break;
    case frame::PassKind::features:
      made(features_pass_, "features").record(stream, address, launch);
      break;
    case frame::PassKind::denoise:
    case frame::PassKind::depth_of_field:
      // The image half's, at the output's size (record_image).
      throw CudaError("Renderer: " + std::string(name(kind)) +
                      " is recorded with the image half");
  }
  (void)begun;
}

void Renderer::record_image(CUstream stream, const FrameSlot& begun,
                            const frame::FrameInputs& inputs,
                            const contracts::Camera& lens,
                            const Prepared& prepared, const FrameLaunch& light,
                            const FrameLaunch& output, RenderTarget target) {
  const auto first = static_cast<std::size_t>(
      std::ranges::find(schedule_.passes, frame::PassKind::denoise) -
      schedule_.passes.begin());
  for (std::size_t p = first; p < schedule_.passes.size(); ++p) {
    const frame::PassKind kind = schedule_.passes[p];
    if (kind == frame::PassKind::denoise) {
      record_denoise(stream, begun, inputs, prepared, light, output);
    } else if (kind == frame::PassKind::depth_of_field) {
      record_depth_of_field(stream, begun, lens, prepared);
    } else {
      record_pass(kind, stream, begun, output, inputs, prepared, target);
    }
  }
}

void Renderer::record_denoise(CUstream stream, const FrameSlot& begun,
                              const frame::FrameInputs& inputs,
                              const Prepared& prepared,
                              const FrameLaunch& light,
                              const FrameLaunch& output) const {
  if (!denoise_ || !denoise_->radiance || !inputs.camera) {
    throw CudaError(
        "Renderer: the denoise pass has no network, radiance image or camera");
  }
  // Its constants: the view axes, the sizes, whether there is a history,
  // and the light's jitter, the frame's: a denoised graph feeds the
  // network, so its frames take it (Metal's denoise_resources()).
  const std::optional<sampling::CameraJitter> jitter =
      frame::camera_jitter(schedule_, inputs.index);
  if (!jitter) {
    throw CudaError("Renderer: a denoised frame with no camera jitter");
  }
  const contracts::DenoiseConstants constants = frame::denoise_constants(
      *inputs.camera, frame::DenoiseFrame{
                          .light = prepared.light,
                          .padded = denoise_->images.padded(),
                          .output = prepared.size,
                          .features = denoise_->features,
                          .history = prepared.denoise_history,
                          .jitter = *jitter,
                      });
  // The light half's features, the slot's, by the light's struct; the
  // slot's full-size albedo by the output's; the hand-off; the radiance
  // image it writes at the output's size.
  denoise_->pass.record(
      stream,
      DenoiseInputs{
          .constants = constants,
          .images = &denoise_->images,
          .light_radiance = &denoise_->images.light_radiance(begun.slot),
          .albedo = light.albedo,
          .normal = light.normal,
          .depth = light.depth,
          .motion = light.motion,
          .full_albedo = output.albedo,
          .full_size = prepared.size,
          .radiance = denoise_->radiance.get(),
      });
}

void Renderer::record_depth_of_field(CUstream stream, const FrameSlot& begun,
                                     const contracts::Camera& lens,
                                     const Prepared& prepared) const {
  if (!depth_of_field_ || !denoise_ || !denoise_->radiance) {
    throw CudaError(
        "Renderer: the depth of field has no denoised radiance to blur");
  }
  // At the output's size, the radiance image the denoise pass wrote, the
  // slot's full-size depth, its constants from the frame's own lens; nothing
  // for a pinhole's frame (passes::blurs), as Metal's.
  const contracts::DepthOfFieldConstants constants =
      passes::depth_of_field_constants(lens, prepared.size);
  if (!passes::blurs(constants)) {
    return;
  }
  const PitchedImage& radiance = *denoise_->radiance;
  depth_of_field_->pass.record(
      stream, DepthOfFieldInputs{
                  .radiance = radiance.address(),
                  .radiance_pitch = radiance.pitch(),
                  .size = prepared.size,
                  .full_depth = denoise_->full_features.at(begun.slot)->depth(),
                  .constants = constants,
                  .images = &depth_of_field_->images,
              });
}

void Renderer::record(const FrameSlot& begun, const frame::FrameInputs& given,
                      RenderTarget target) {
  const frame::Extent size = target.size;
  // The frame as its light is traced: through a pinhole in a graph that
  // feeds the network (core/frame/schedule.h, traced_inputs), as Metal's.
  const frame::FrameInputs inputs = frame::traced_inputs(schedule_, given);
  if (!prepared_ || prepared_->index != inputs.index ||
      !(prepared_->size == size)) {
    throw CudaError("Renderer::record: frame " + std::to_string(inputs.index) +
                    " was not prepared at this size (Renderer::prepare)");
  }
  // Two parts for a graph of one is refused; one part for a denoised
  // graph records both halves on the light stream (renderer.h).
  if (begun.parts == Parts::two && parts() == Parts::one) {
    throw CudaError(
        "Renderer::record: a frame of two parts for a graph of one "
        "(Renderer::parts)");
  }
  const Prepared prepared = *prepared_;
  prepared_.reset();
  // A temporal graph merges last frame when the history holds it and the
  // renderer recorded it (its camera kept): never at a history's first.
  const bool previous = reuses_history_ && prepared.reused_frames > 0 &&
                        previous_camera_.has_value();

  animate(begun, inputs, previous);
  if (non_finite_) {
    non_finite_->begin_frame(begun);
  }
  if (mnee_) {
    mnee_->lost.begin_frame(begun);
  }
  // The frame's launch parameters, uploaded once for every pass; a
  // denoised frame's output struct beside them (renderer.h).
  const FrameLaunch launch =
      frame_launch(begun, inputs, prepared, prepared.light, previous);
  std::memcpy(launch_.bytes(begun.slot).data(), &launch, sizeof(launch));
  launch_.upload(begun);
  FrameLaunch output{};
  if (denoise_) {
    output = output_launch(begun, inputs, launch, size);
    std::memcpy(output_launch_->bytes(begun.slot).data(), &output,
                sizeof(output));
    output_launch_->upload(begun);
  }
  // The light half: every pass before the denoise pass; in a graph that
  // feeds the network the fill before the presenting pass, or, denoised,
  // the full-size albedo and the fill after the light's passes.
  const auto image_at = static_cast<std::size_t>(
      std::ranges::find(schedule_.passes, frame::PassKind::denoise) -
      schedule_.passes.begin());
  const std::size_t fill_at = !feeds_network_ ? schedule_.passes.size()
                              : denoise_      ? image_at
                                              : schedule_.passes.size() - 1;
  for (std::size_t p = 0; p < image_at; ++p) {
    if (p == fill_at) {
      made(fill_, "checkerboard fill")
          .record(begun.stream, launch_.address(), launch);
    }
    record_pass(schedule_.passes[p], begun.stream, begun, launch, inputs,
                prepared, target);
  }
  if (denoise_) {
    // The full-size albedo, the features pass at the output's size; then
    // the fill, which writes the slot's hand-off.
    made(features_pass_, "features")
        .record(begun.stream, output_launch_->address(), output);
    made(fill_, "checkerboard fill")
        .record(begun.stream, launch_.address(), launch);
  }
  if (non_finite_) {
    non_finite_->end_frame(begun);
  }
  if (mnee_) {
    mnee_->lost.end_frame(begun);
  }
  if (denoise_) {
    // The image half: on the image stream after the hand-off in two
    // parts, after the light half on its stream in one.
    if (!given.camera) {
      throw CudaError("Renderer::record: a denoised frame has no camera");
    }
    const CUstream image =
        begun.parts == Parts::two ? submission_.hand_off() : begun.stream;
    record_image(image, begun, inputs, *given.camera, prepared, launch, output,
                 target);
  }
  // What the next frame's temporal merge asks of this one; what
  // read_network_input() reads of it.
  previous_camera_ = inputs.camera;
  previous_time_ = inputs.time;
  last_slot_ = begun.slot;
  last_jitter_ = frame::camera_jitter(schedule_, inputs.index)
                     .value_or(sampling::CameraJitter{});
  if (features_) {
    features_->last_slot = begun.slot;
  }
}

std::uint64_t Renderer::non_finite_samples() const noexcept {
  return non_finite_ ? non_finite_->count() : 0u;
}

std::uint64_t Renderer::mnee_lost_requests() const noexcept {
  return mnee_ ? mnee_->lost.count() : 0u;
}

void Renderer::read_accumulated(std::span<float> out) const {
  const PitchedImage* image = shown();
  if (image == nullptr) {
    throw CudaError(
        "Renderer::read_accumulated: the graph keeps no accumulated image, "
        "or none has been rendered");
  }
  image->read(out);
}

FeatureImages::Read Renderer::read_features() const {
  if (!features_) {
    throw CudaError("Renderer::read_features: the graph has no features pass");
  }
  return features_->images(features_->last_slot).read();
}

Renderer::NetworkInput Renderer::read_network_input() const {
  if (!checkerboard_ || !last_slot_) {
    throw CudaError(
        "Renderer::read_network_input: the frame graph's light is not the "
        "network's input, or no frame has been recorded");
  }
  const PitchedImage& image = denoise_
                                  ? denoise_->images.light_radiance(*last_slot_)
                                  : *checkerboard_->filled();
  NetworkInput out{
      image.size(),
      std::vector<float>(std::size_t{image.size().width} * image.size().height *
                         contracts::accumulated_image_channels),
      last_jitter_};
  image.read(out.rgba);
  return out;
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
  renderer.prepare(inputs, target.size());
  const FrameSlot begun = submission.begin(renderer.parts());
  renderer.record(begun, inputs, target.target());
  submission.commit();
  return begun.sequence;
}

std::uint64_t render_to_window(Submission& submission, Presenter& presenter,
                               Renderer& renderer,
                               const frame::FrameInputs& inputs,
                               const WindowImage& image) {
  renderer.prepare(inputs, presenter.size());
  const FrameSlot begun = submission.begin(renderer.parts());
  if (begun.slot != image.slot) {
    throw CudaError("render_to_window: frame " +
                    std::to_string(begun.sequence) +
                    " takes another slot than its image was acquired for");
  }
  // The target is written by the frame's last half: the image stream's in
  // two parts, which the hand-off joins after this wait (submission.h,
  // image_stream), the light stream's in one; its last work is there too.
  const CUstream target_stream =
      begun.parts == Parts::two ? submission.image_stream() : begun.stream;
  presenter.wait_for_slot(target_stream, image);
  renderer.record(begun, inputs, image.target);
  presenter.present(target_stream, image, begun.sequence);
  submission.commit();
  return begun.sequence;
}

}  // namespace serenity::cuda
