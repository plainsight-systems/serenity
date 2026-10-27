#include "metal/frame/renderer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/animation/animate.h"
#include "core/camera/thin_lens.h"
#include "core/contracts/camera.h"
#include "core/contracts/checkerboard.h"
#include "core/contracts/denoise.h"
#include "core/contracts/depth_of_field.h"
#include "core/contracts/features.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/denoise/network.h"
#include "core/frame/frame_constants.h"
#include "core/frame/refusals.h"
#include "core/passes/depth_of_field.h"
#include "core/sampling/pairing.h"
#include "core/shapes/mesh_levels.h"
#include "metal/device/error.h"
#include "metal/device/readback.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "serenity/metallib/shaders.h"

namespace serenity::metal {

namespace {

// Where in a slot of the ring each contract is (renderer.h): each bound as
// `constant`, so on Metal's constant offset alignment (support.h) and its
// type's, with room to grow. A slot is one FrameArray copy,
// buffer_alignment bytes.
constexpr std::size_t constants_offset = 0;
constexpr std::size_t camera_offset = 128;
constexpr std::size_t ring_slot_size = buffer_alignment;
static_assert(constants_offset + sizeof(contracts::FrameConstants) <=
              camera_offset);
static_assert(camera_offset + sizeof(contracts::CameraData) <= ring_slot_size);
static_assert(constants_offset % constant_offset_alignment == 0 &&
              camera_offset % constant_offset_alignment == 0);
static_assert(camera_offset % alignof(contracts::CameraData) == 0);
// Both are written into the ring as their bytes (SL.con.4, COPY.6).
static_assert(std::is_trivially_copyable_v<contracts::FrameConstants>);
static_assert(std::is_trivially_copyable_v<contracts::CameraData>);

// What `held` holds: a family, setting or input the graph's passes have
// (renderer.h), made or given before any frame reads it. One that is not
// there is this renderer's defect, refused by name (E.2, I.6), not read.
template <typename T>
T& made(std::optional<T>& held, std::string_view what) {
  if (!held) {
    throw MetalError("Renderer: no " + std::string(what) +
                     ", which the frame graph's passes need");
  }
  return *held;
}
template <typename T>
const T& made(const std::optional<T>& held, std::string_view what) {
  if (!held) {
    throw MetalError("Renderer: no " + std::string(what) +
                     ", which the frame graph's passes need");
  }
  return *held;
}

// The run's path reuse settings: the graph's, the checkerboard on where
// the network's input is traced checkerboarded. The ReSTIR PT passes'
// pipelines are specialized by them (passes/restir_pt/reuse_constants.h)
// and its arrays sized by them, one derivation for both (ES.3). Throws
// MetalError if the graph has no ReSTIR PT.
contracts::PathReuseSettings run_reuse_settings(const frame::Schedule& schedule,
                                                bool feeds_network) {
  contracts::PathReuseSettings settings =
      made(schedule.restir_pt, "ReSTIR PT settings");
  settings.checkerboard = feeds_network ? 1u : 0u;
  return settings;
}

// A pass recorded into a compute encoder, as every pass but the denoise
// pass is; that one records with its command buffer, its network in an
// encoder of its own (record_image()), and is refused here (I.5).
template <typename P>
void record_pass(const P& pass, MTL4::ComputeCommandEncoder* encoder,
                 const FrameResources& resources) {
  pass.record(encoder, resources);
}
void record_pass(const ShadedInMerge& /*pass*/,
                 MTL4::ComputeCommandEncoder* /*encoder*/,
                 const FrameResources& /*resources*/) {
  throw MetalError(
      "Renderer: the shade in the spatial merge records nothing; "
      "record_light() skips its step");
}
void record_pass(const DenoisePass& /*pass*/,
                 MTL4::ComputeCommandEncoder* /*encoder*/,
                 const FrameResources& /*resources*/) {
  throw MetalError(
      "Renderer: the denoise pass records with its command buffer "
      "(record_image())");
}

// The argument table's sizes (renderer.h). Each family of passes binds at
// most its constant's buffers and textures through the table
// (metal/passes/bindings.h), each the index past its passes' last, asserted
// below rather than restated in words; a graph's table holds the most of
// the families it has (buffer_bindings()). A new kind of material,
// texture, medium or light adds to the scene's block, not here.
//
// Every graph: the path pass, the features pass, the presenting pass.
constexpr NS::UInteger max_buffers = 9;
constexpr NS::UInteger max_textures = 5;
static_assert(bindings::path::non_finite < max_buffers);
static_assert(bindings::features::mesh_block < max_buffers);
static_assert(bindings::tone_map::target < max_textures);
static_assert(bindings::features::visibility < max_textures);
// A graph with the denoise pass.
constexpr NS::UInteger max_denoise_buffers = 12;
constexpr NS::UInteger max_denoise_textures = 8;
static_assert(bindings::denoise::coarse < max_denoise_buffers);
static_assert(bindings::denoise::radiance < max_denoise_textures);
// The checkerboard's fill, a denoised graph's.
static_assert(bindings::checkerboard_fill::fill_constants <
              max_denoise_buffers);
static_assert(bindings::checkerboard_fill::history < max_denoise_textures);
// The depth of field, a denoised graph's.
static_assert(bindings::depth_of_field::dilated < max_denoise_buffers);
static_assert(bindings::depth_of_field::near_filtered < max_denoise_textures);
// A ReSTIR PT graph (its initial pass's work list and the MNEE pass bind
// within it).
constexpr NS::UInteger max_restir_pt_buffers = 31;  // Metal 4's most
static_assert(bindings::restir_pt::mnee_header < max_restir_pt_buffers);
static_assert(bindings::restir_pt::last_light_power < max_restir_pt_buffers &&
              bindings::restir_pt::reconnects < max_restir_pt_buffers);
static_assert(bindings::mnee::lost < max_restir_pt_buffers &&
              bindings::mnee::converged_count < max_restir_pt_buffers);

// Whether `schedule` has a pass of `kind`.
bool has(const frame::Schedule& schedule, frame::PassKind kind) {
  return std::ranges::find(schedule.passes, kind) != schedule.passes.end();
}

// The buffers `schedule`'s table holds: the most of its families' (above).
NS::UInteger buffer_bindings(const frame::Schedule& schedule) {
  NS::UInteger most = max_buffers;
  if (frame::denoises(schedule)) {
    most = std::max(most, max_denoise_buffers);
  }
  if (std::ranges::any_of(schedule.passes, frame::is_restir_pt)) {
    most = std::max(most, max_restir_pt_buffers);
  }
  return most;
}

// The textures it holds.
// The textures it holds: a denoised graph's, or the fill's in any graph
// whose light is the network's input.
NS::UInteger texture_bindings(const frame::Schedule& schedule) {
  return frame::denoises(schedule) || frame::feeds_network(schedule)
             ? max_denoise_textures
             : max_textures;
}

// A refusal the core gives (core/frame/refusals.h), thrown as this
// backend's error: `prefix` and the reason. None, nothing.
void refuse(const std::optional<std::string>& reason,
            std::string_view prefix = "Renderer: ") {
  if (reason) {
    throw MetalError(std::string(prefix) + *reason);
  }
}

NS::SharedPtr<MTL4::ArgumentTable> make_argument_table(MTL::Device* device,
                                                       NS::UInteger buffers,
                                                       NS::UInteger textures) {
  const auto pool = scoped_pool();  // Metal's error, if any, is autoreleased
  auto descriptor =
      NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
  descriptor->setMaxBufferBindCount(buffers);
  descriptor->setMaxTextureBindCount(textures);
  NS::Error* error = nullptr;
  auto table =
      NS::TransferPtr(device->newArgumentTable(descriptor.get(), &error));
  if (!table) {
    throw MetalError("Renderer: the device made no argument table: " +
                     describe(error));
  }
  return table;
}

// The features pass's constants for a frame of `size` (contract 17) into
// `ring`'s `slot`: whether there is a frame before, and its camera framed
// for `size`. Their address. For the features at the light size and the
// denoiser's full-size albedo alike.
MTL::GPUAddress write_feature_constants(
    FrameArray& ring, std::uint32_t slot,
    const std::optional<frame::PreviousFrame>& previous, frame::Extent size) {
  const contracts::FeatureConstants features =
      frame::feature_constants(previous, size);
  std::memcpy(ring.bytes(slot).data(), &features, sizeof(features));
  return ring.address(slot);
}

}  // namespace

// ReSTIR PT's (renderer.h): its arrays, its settings and reuse textures,
// and with temporal reuse the ring of its reuse constants.
Renderer::RestirPtFamily::RestirPtFamily(
    const Device& device, Submission& submission,
    const contracts::PathReuseSettings& settings,
    std::span<const std::uint16_t> textures, bool spatial, bool temporal,
    std::uint32_t lights)
    : images(device, submission,
             spatial ? PathReuseImages::Spatial::merged
                     : PathReuseImages::Spatial::none,
             temporal ? PathReuseImages::Temporal::merged
                      : PathReuseImages::Temporal::none,
             settings.checkerboard != 0u ? PathReuseImages::Grid::checkerboard
                                         : PathReuseImages::Grid::image,
             lights),
      statics(
          device, submission,
          std::array<std::span<const std::byte>, 1>{std::as_bytes(textures)}) {
  if (temporal) {
    constants.emplace(
        device, submission,
        std::vector<std::byte>(sizeof(contracts::PathReuseConstants)),
        FrameArray::Copies::per_frame);
  }
}

void Renderer::RestirPtFamily::bind(
    FrameResources& resources, const FrameSlot& begun, std::uint64_t index,
    const std::optional<contracts::Camera>& previous_camera,
    frame::Extent light) {
  const PathReuseImages::Addresses arrays = images.addresses(index);
  resources.pairing = statics.address(0);
  resources.path_reservoirs = arrays.reservoirs;
  resources.primaries = arrays.primaries;
  resources.light = arrays.light;
  resources.shifts = arrays.shifts;
  resources.reconnects = arrays.reconnects;
  resources.replays = arrays.replays;
  resources.pair_counts = arrays.counts;
  resources.pair_grids = arrays.grids;
  resources.light_tiles = arrays.light_tiles;
  resources.light_power = arrays.light_power;
  resources.last_light_power = arrays.last_light_power;
  resources.last_primaries = arrays.last_primaries;
  resources.path_finals = arrays.finals;
  resources.last_path_finals = arrays.last_finals;
  resources.previous_positions = arrays.previous_positions;
  resources.previous_pixels = arrays.previous_pixels;
  resources.duplication = arrays.duplication;
  if (constants) {
    // Whether there is a frame before to merge, with its camera framed
    // for this frame's size.
    const contracts::PathReuseConstants reuse =
        frame::path_reuse_constants(previous_camera, light);
    std::memcpy(constants->bytes(begun.slot).data(), &reuse, sizeof(reuse));
    resources.path_constants = constants->address(begun.slot);
  }
}

// MNEE's (renderer.h): its work list and its count of lost requests.
Renderer::MneeFamily::MneeFamily(const Device& device, Submission& submission)
    : list(device, submission),
      lost(device, submission, "MNEE requests lost to a full work list") {}

void Renderer::MneeFamily::bind(FrameResources& resources,
                                const FrameSlot& begun) {
  const MneeList::Addresses addresses = list.addresses();
  resources.mnee_requests = addresses.requests;
  resources.mnee_header = addresses.header;
  resources.mnee_grid = addresses.grid;
  resources.mnee_walks = addresses.walks;
  resources.mnee_converged = addresses.converged;
  resources.mnee_walk_header = addresses.walk_header;
  lost.begin_frame(begun.slot, begun.sequence);
  resources.mnee_lost_counter = lost.address(begun.slot);
}

// The features pass's (renderer.h): its images, its constants' ring and,
// when shapes move, the shapes as the frame before placed them.
Renderer::FeatureFamily::FeatureFamily(const Device& device,
                                       Submission& submission,
                                       const scene::SceneDescription& scene,
                                       bool moves, Sets kept)
    : constants(device, submission,
                std::vector<std::byte>(sizeof(contracts::FeatureConstants)),
                FrameArray::Copies::per_frame) {
  sets[0].emplace(device, submission);
  if (kept == Sets::per_slot) {
    for (std::size_t slot = 1; slot < sets.size(); ++slot) {
      sets.at(slot).emplace(device, submission);
    }
  }
  if (moves) {
    previous_transforms.emplace(device, submission, scene.shapes.transforms,
                                FrameArray::Copies::per_frame);
  }
}

void Renderer::FeatureFamily::prepare(frame::Extent light) {
  for (std::optional<FeatureImages>& set : sets) {
    if (set) {
      set->prepare(light);
    }
  }
}

const FeatureImages& Renderer::FeatureFamily::images(std::uint32_t slot) const {
  // One set per slot when handed off, else the one (renderer.h).
  return made(sets[1] ? sets.at(slot) : sets[0], "feature images");
}

void Renderer::FeatureFamily::bind(
    FrameResources& resources, const FrameSlot& begun,
    const std::optional<frame::PreviousFrame>& previous, frame::Extent light) {
  // Whether there is a frame before, and its camera framed for this
  // frame's size; its shapes, animated in record() where they move.
  resources.feature_constants =
      write_feature_constants(constants, begun.slot, previous, light);
  resources.feature_last_transforms =
      previous_transforms ? previous_transforms->address(begun.slot)
                          : resources.transforms;
  const FeatureImages& written = images(begun.slot);
  resources.albedo = written.albedo();
  resources.normal = written.normal();
  resources.depth = written.depth();
  resources.motion = written.motion();
  resources.visibility = written.visibility();
  resources.raster_depth = written.raster_depth();
  last_slot = begun.slot;
}

// The denoise pass's (renderer.h): the network's last width, its buffers,
// the target-size ring, its constants' ring, and the full-size albedo's
// images, constants and kernel.
Renderer::DenoiseFamily::DenoiseFamily(const Device& device,
                                       Submission& submission,
                                       const Library& library,
                                       const denoise::Network& network)
    : features(network.widths.dec.back()),
      images(device, submission),
      output_constants(device, submission,
                       std::vector<std::byte>(ring_slot_size),
                       FrameArray::Copies::per_frame),
      constants(device, submission,
                std::vector<std::byte>(sizeof(contracts::DenoiseConstants)),
                FrameArray::Copies::per_frame),
      full_feature_constants(
          device, submission,
          std::vector<std::byte>(sizeof(contracts::FeatureConstants)),
          FrameArray::Copies::per_frame),
      full_features_pass(device, library) {
  for (std::optional<FeatureImages>& set : full_features) {
    set.emplace(device, submission);
  }
}

// Checkerboard rendering's (renderer.h): the fill, its constants' ring and
// its history.
Renderer::CheckerboardFamily::CheckerboardFamily(
    const Device& device, Submission& submission, const Library& library,
    CheckerboardImages::Filled filled)
    : fill(library),
      constants(
          device, submission,
          std::vector<std::byte>(sizeof(contracts::CheckerboardConstants)),
          FrameArray::Copies::per_frame),
      images(device, submission, filled) {}

// The depth of field's (renderer.h): its images and its constants' ring.
Renderer::DepthOfFieldFamily::DepthOfFieldFamily(const Device& device,
                                                 Submission& submission)
    : images(device, submission),
      constants(
          device, submission,
          std::vector<std::byte>(sizeof(contracts::DepthOfFieldConstants)),
          FrameArray::Copies::per_frame) {}

Renderer::Renderer(const Device& device, Submission& submission,
                   const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : submission_(submission),
      library_(device, serenity::metallib::shaders),
      constants_(device, submission, std::vector<std::byte>(ring_slot_size),
                 FrameArray::Copies::per_frame),
      arguments_(make_argument_table(device.handle(), buffer_bindings(schedule),
                                     texture_bindings(schedule))),
      denoises_(frame::denoises(schedule)),
      feeds_network_(frame::feeds_network(schedule)),
      schedule_(schedule),
      needs_scene_(std::ranges::any_of(schedule.passes, frame::needs_scene)) {
  // The core decides which schedules can be carried out (principle 10).
  refuse(frame::invalid(schedule));
  refuse(frame::scene_refusal(schedule, scene));
  const auto pool = scoped_pool();
  // The network, read once (core/denoise/network.h); its weights go to the
  // GPU with its pass, its widths size its buffers.
  std::optional<denoise::Network> network;
  if (schedule.denoise) {
    network = denoise::load_network(schedule.denoise->weights);
  }
  make_steps(device, schedule, network);
  make_images(device, schedule);
  make_restir_pt(device, schedule, scene);
  make_mnee(device, schedule, scene);
  if (needs_scene_) {
    make_scene(device, *scene);
  }
  // The features pass's. Its pass reads a scene, which the check above
  // has made sure of.
  if (frame::writes_features(schedule)) {
    // One set per frame slot when the denoiser reads them from the
    // other queue (renderer.h, "Two halves").
    features_.emplace(
        device, submission_, *scene, animation::moves(animation_),
        denoises_ ? FeatureFamily::Sets::per_slot : FeatureFamily::Sets::one);
  }
  // The denoise pass's. The core has made sure the graph has the features
  // pass and a light pass before it (frame::invalid).
  if (denoises_) {
    denoise_.emplace(device, submission_, library_,
                     made(network, "denoiser network"));
  }
  // Checkerboard rendering's, a graph's whose light is the network's input
  // (frame::feeds_network), its own filled light without the denoise pass.
  if (feeds_network_) {
    checkerboard_.emplace(device, submission_, library_,
                          denoises_ ? CheckerboardImages::Filled::none
                                    : CheckerboardImages::Filled::kept);
  }
  // The depth of field's, a denoised graph's (frame::invalid).
  if (frame::lens_after_network(schedule)) {
    depth_of_field_.emplace(device, submission_);
  }
}

// Each of the schedule's pass kinds as this backend's pass for it, in
// schedule order (renderer.h); and the first pass that writes the images
// between passes, accumulates, or writes ReSTIR PT's arrays, which the
// frame before's passes read, or the feature images, which the frame
// before's readers read.
void Renderer::make_steps(const Device& device, const frame::Schedule& schedule,
                          const std::optional<denoise::Network>& network) {
  const auto crosses_frames = [](frame::PassKind kind) {
    return frame::writes_radiance(kind) || frame::accumulates(kind) ||
           frame::is_restir_pt(kind) || frame::writes_features(kind) ||
           kind == frame::PassKind::denoise;
  };
  if (const auto first = std::ranges::find_if(schedule.passes, crosses_frames);
      first != schedule.passes.end()) {
    first_cross_frame_ =
        static_cast<std::size_t>(first - schedule.passes.begin());
  }
  // The image half begins at the denoise pass (renderer.h, "Two halves");
  // the core puts every pass that crosses frames on the light's side of it
  // (frame::invalid: the features and a light pass come before it).
  image_begins_ = static_cast<std::size_t>(
      std::ranges::find(schedule.passes, frame::PassKind::denoise) -
      schedule.passes.begin());
  // The fill, in a graph that feeds the network: before the denoise pass,
  // or without it before the presenting pass, the last (frame::invalid).
  fill_at_ = !frame::feeds_network(schedule) ? schedule.passes.size()
             : frame::denoises(schedule)     ? image_begins_
                                             : schedule.passes.size() - 1u;
  steps_.reserve(schedule.passes.size());
  for (frame::PassKind kind : schedule.passes) {
    // No default: a kind this backend does not implement fails the build
    // (core/frame/schedule.h).
    switch (kind) {
      case frame::PassKind::test_pattern:
        steps_.push_back(
            Step{kind, Pass{std::in_place_type<TestPatternPass>, library_}});
        break;
      case frame::PassKind::path:
        steps_.push_back(
            Step{kind, Pass{std::in_place_type<PathPass>, library_}});
        break;
      case frame::PassKind::tone_map:
        // The schedule has its settings with its pass (frame::invalid).
        steps_.push_back(Step{
            kind,
            Pass{std::in_place_type<ToneMapPass>, device, library_, submission_,
                 made(schedule.tone_map, "tone map settings")}});
        break;
      case frame::PassKind::restir_pt_initial:
        steps_.push_back(Step{
            kind,
            Pass{std::in_place_type<RestirPtInitialPass>, library_, submission_,
                 run_reuse_settings(schedule, feeds_network_)}});
        break;
      case frame::PassKind::restir_pt_spatial:
        steps_.push_back(
            Step{kind, Pass{std::in_place_type<RestirPtSpatialPass>, library_,
                            run_reuse_settings(schedule, feeds_network_)}});
        break;
      case frame::PassKind::restir_pt_shade:
        // The spatial pass's merge shades when the graph has it
        // (ShadedInMerge, renderer.h).
        if (has(schedule, frame::PassKind::restir_pt_spatial)) {
          steps_.push_back(Step{kind, Pass{ShadedInMerge{}}});
        } else {
          steps_.push_back(
              Step{kind, Pass{std::in_place_type<RestirPtShadePass>, library_,
                              run_reuse_settings(schedule, feeds_network_)}});
        }
        break;
      case frame::PassKind::restir_pt_temporal:
        steps_.push_back(
            Step{kind, Pass{std::in_place_type<RestirPtTemporalPass>, library_,
                            run_reuse_settings(schedule, feeds_network_)}});
        break;
      case frame::PassKind::restir_pt_duplication:
        steps_.push_back(Step{
            kind, Pass{std::in_place_type<RestirPtDuplicationPass>, library_,
                       run_reuse_settings(schedule, feeds_network_)}});
        break;
      case frame::PassKind::features:
        steps_.push_back(Step{
            kind, Pass{std::in_place_type<FeaturesPass>, device, library_}});
        break;
      case frame::PassKind::denoise:
        // The schedule has its settings with its pass (frame::invalid).
        steps_.push_back(
            Step{kind, Pass{std::in_place_type<DenoisePass>, device, library_,
                            submission_, made(network, "denoiser network")}});
        break;
      case frame::PassKind::depth_of_field:
        steps_.push_back(
            Step{kind, Pass{std::in_place_type<DepthOfFieldPass>, library_}});
        break;
    }
  }
}

// The images between passes and the accumulated image (renderer.h). The
// pass that accumulates is the one that writes radiance (the core allows
// one of each, and path is both), and its accumulated image is the frame's
// radiance (passes/path/path.h): no image of its own then. In a denoised
// graph the radiance is the denoise pass's, at the target's size, and the
// accumulated image the light's, at the light size.
void Renderer::make_images(const Device& device,
                           const frame::Schedule& schedule) {
  const bool accumulates = frame::accumulates(schedule);
  const bool radiance =
      std::ranges::any_of(schedule.passes, frame::writes_radiance);
  const FrameImages::Radiance own = accumulates && !denoises_
                                        ? FrameImages::Radiance::accumulated
                                        : FrameImages::Radiance::image;
  const FrameImages::Bloom bloom = schedule.tone_map
                                       ? FrameImages::Bloom::pyramid
                                       : FrameImages::Bloom::none;
  if (radiance && (own == FrameImages::Radiance::image ||
                   bloom == FrameImages::Bloom::pyramid)) {
    images_ = std::make_unique<FrameImages>(device, submission_, own, bloom);
  }
  if (accumulates) {
    accumulation_ = std::make_unique<Accumulation>(device, submission_);
    non_finite_ = std::make_unique<FailureCount>(device, submission_,
                                                 "samples not finite");
  }
}

// ReSTIR PT's, only for its graph (renderer.h).
void Renderer::make_restir_pt(const Device& device,
                              const frame::Schedule& schedule,
                              const scene::SceneDescription* scene) {
  if (!schedule.restir_pt) {
    return;
  }
  // What the G-buffer and the light tiles pack (contracts/primary_hit.h,
  // light_tile.h), refused rather than wrapped (I.5).
  refuse(frame::restir_pt_refusal(*scene));
  const bool temporal = has(schedule, frame::PassKind::restir_pt_temporal);
  reuses_history_ = temporal;
  const bool spatial = has(schedule, frame::PassKind::restir_pt_spatial);
  const contracts::PathReuseSettings settings =
      run_reuse_settings(schedule, feeds_network_);
  // With spatial reuse, the three reuse textures, built once on the CPU
  // from the graph's sigma and a fixed seed (core/sampling/pairing.h); one
  // zero texel without. With the checkerboard the pairing is over the
  // traced lattice, so at sigma over its step, sigma pixels along each
  // axis (renderer.h; the core has refused a sigma the lattice cannot
  // take, frame::invalid).
  const double sigma =
      sampling::reuse_texture_sigma(settings.sigma, feeds_network_);
  const std::vector<std::uint16_t> textures =
      spatial ? sampling::reuse_textures(sigma, sampling::reuse_texture_seed)
              : std::vector<std::uint16_t>(1, 0u);
  restir_pt_.emplace(device, submission_, settings, textures, spatial, temporal,
                     scene->light_counts.lights);
}

// MNEE's work list, a ReSTIR PT graph's (its requests add to the light
// buffer). Its requests keep a shape in 16 bits (contract 16), refused
// rather than wrapped (I.5).
void Renderer::make_mnee(const Device& device, const frame::Schedule& schedule,
                         const scene::SceneDescription* scene) {
  if (!schedule.restir_pt) {
    return;
  }
  refuse(frame::mnee_refusal(*scene));
  mnee_.emplace(device, submission_);
}

// The scene on the GPU, for a graph that reads it, and what its frames
// place (renderer.h).
void Renderer::make_scene(const Device& device,
                          const scene::SceneDescription& scene) {
  scene_ = std::make_unique<SceneBuffers>(device, submission_, scene);
  animation_ = scene.animation;
  const bool moves = animation::moves(animation_);
  transforms_ = std::make_unique<ShapeTransforms>(
      device, submission_, scene.shapes.transforms,
      moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
  // Checked here, not by animate() in a frame, so nothing about the scene
  // can fail once a frame's submission has begun (renderer.h).
  refuse(frame::glower_refusal(scene));
  glows_ = std::make_unique<LightGlows>(device, submission_,
                                        scene.light_counts.spheres,
                                        !animation_.glowers.empty());
  // Last frame's, for a temporal graph whose scene changes (renderer.h).
  if (reuses_history_ && animation::changes(animation_)) {
    last_transforms_ = std::make_unique<ShapeTransforms>(
        device, submission_, scene.shapes.transforms,
        moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
    last_glows_ = std::make_unique<LightGlows>(device, submission_,
                                               scene.light_counts.spheres,
                                               !animation_.glowers.empty());
  }
  // Where the families meet: the acceleration structure learns which
  // shapes move, by the movers' targets, and which are lights, by the
  // scene's answer for each shape (contract 3), and nothing else of
  // either: the lights are the shapes a ray carrying the occluders skips
  // (contract 19).
  std::vector<std::uint32_t> moving;
  moving.reserve(animation_.movers.size());
  for (const animation::Mover& mover : animation_.movers) {
    moving.push_back(mover.target);
  }
  const std::vector<std::uint32_t> lights = scene::light_shapes(scene);
  // Each sphere's mesh, by its error (core/shapes/mesh_levels.h), a scene
  // whose sphere no level serves refused in the core's words.
  refuse(shapes::sphere_level_refusal(scene));
  const std::vector<std::uint8_t> levels = shapes::sphere_levels(scene);
  acceleration_ = std::make_unique<SceneAcceleration>(
      device, submission_, scene.shapes,
      SceneAcceleration::Roles{
          .moving = moving, .lights = lights, .levels = levels});
}

Renderer::~Renderer() {
  // The pipelines and the argument table are not in the residency set, so
  // no Resident waits for them: this does, before any member is released.
  submission_.wait_idle();
}

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
  if (!prepares()) {
    return;
  }
  prepared_.reset();
  // Every image before the denoise pass at the light size; the target's
  // own in a graph without it (frame::light_size).
  const frame::Extent light = frame::light_size(schedule_, size);
  if (images_) {
    images_->prepare(size);
  }
  if (features_) {
    features_->prepare(light);
  }
  if (denoise_) {
    for (std::optional<FeatureImages>& set : denoise_->full_features) {
      made(set, "full-size features").prepare(size);
    }
  }
  const bool fill_history =
      checkerboard_ ? checkerboard_->images.prepare(inputs, light) : false;
  if (depth_of_field_) {
    depth_of_field_->images.prepare(size);
  }
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
  // The denoise pass's: its images at the network's rung for the light,
  // then the network's memory at it (passes/denoise/denoise.h), after the
  // images, which drain the frames in flight at a new size.
  bool denoise_history = false;
  for (Step& step : steps_) {
    if (step.kind == frame::PassKind::denoise) {
      auto& pass = std::get<DenoisePass>(step.pass);
      const frame::Extent grid = pass.grid(light);
      denoise_history =
          made(denoise_, "denoiser").images.prepare(inputs, light, grid);
      pass.prepare(grid);
    }
  }
  prepared_ =
      Prepared{inputs.index, size, held, reused, denoise_history, fill_history};
}

bool Renderer::prepares() const noexcept {
  return accumulation_ || images_ || restir_pt_ || mnee_ || features_ ||
         denoise_ || depth_of_field_;
}

std::uint64_t Renderer::non_finite_samples() const noexcept {
  return non_finite_ ? non_finite_->count() : 0u;
}

std::uint64_t Renderer::mnee_lost_requests() const noexcept {
  return mnee_ ? mnee_->lost.count() : 0u;
}

void Renderer::read_accumulated(std::span<float> out) {
  MTL::Texture* image = nullptr;
  if (denoises_) {
    image = images_ ? images_->radiance()
                    : nullptr;  // the denoised radiance, at the target's size
  } else if (checkerboard_) {
    image =
        checkerboard_->images
            .filled();  // the fill's light, which the graph shows (renderer.h)
  } else if (accumulation_) {
    image = accumulation_->texture();
  }
  if (image == nullptr) {
    throw MetalError(
        "read_accumulated: the frame graph accumulates nothing, or no frame "
        "has made its image");
  }
  // RGBA32Float
  constexpr std::size_t channels = contracts::accumulated_image_channels;
  const std::size_t row_bytes =
      std::size_t{image->width()} * channels * sizeof(float);
  const std::size_t floats =
      std::size_t{image->width()} * image->height() * channels;
  if (out.size() != floats) {
    throw MetalError("read_accumulated: " + std::to_string(out.size()) +
                     " floats given for an image of " + std::to_string(floats));
  }
  const frame::Extent size{static_cast<std::uint32_t>(image->width()),
                           static_cast<std::uint32_t>(image->height())};
  const std::array<ImageCopy, 1> copy = {ImageCopy{image, 0, row_bytes}};
  const auto buffer =
      read_back(image->device(), submission_, size, row_bytes * size.height,
                copy, "read_accumulated");
  std::memcpy(out.data(), buffer->contents(), floats * sizeof(float));
}

Renderer::NetworkInput Renderer::read_network_input() {
  if (!checkerboard_ || !last_slot_) {
    throw MetalError(
        "read_network_input: the frame graph's light is not the network's "
        "input, or no frame has "
        "been recorded");
  }
  MTL::Texture* image = denoise_ ? denoise_->images.light_radiance(*last_slot_)
                                 : checkerboard_->images.filled();
  // RGBA32Float, as the hand-off's
  constexpr std::size_t channels = contracts::accumulated_image_channels;
  const frame::Extent size{static_cast<std::uint32_t>(image->width()),
                           static_cast<std::uint32_t>(image->height())};
  const std::size_t row_bytes =
      std::size_t{size.width} * channels * sizeof(float);
  const std::array<ImageCopy, 1> copy = {ImageCopy{image, 0, row_bytes}};
  const auto buffer =
      read_back(image->device(), submission_, size, row_bytes * size.height,
                copy, "read_network_input");
  NetworkInput out{
      size,
      std::vector<float>(std::size_t{size.width} * size.height * channels),
      last_jitter_};
  std::memcpy(out.rgba.data(), buffer->contents(),
              out.rgba.size() * sizeof(float));
  return out;
}

FeatureImages::Read Renderer::read_features() const {
  if (!features_ ||
      features_->images(features_->last_slot).albedo() == nullptr) {
    throw MetalError(
        "read_features: the frame graph has no features pass, or no frame has "
        "made its images");
  }
  return features_->images(features_->last_slot).read();
}

// What prepare() readied for the frame: checked to be this frame at this
// size where the graph prepares anything, and taken (renderer.h).
Renderer::Prepared Renderer::take_prepared(const frame::FrameInputs& inputs,
                                           frame::Extent size) {
  if (!prepares()) {
    return Prepared{inputs.index, size, 0u, 0u, false, false};
  }
  if (!prepared_ || prepared_->index != inputs.index ||
      !(prepared_->size == size)) {
    throw MetalError("Renderer::record: frame " + std::to_string(inputs.index) +
                     " was not prepared at this size (Renderer::prepare)");
  }
  const Prepared taken = *prepared_;
  prepared_.reset();
  return taken;
}

// The frame's constants and its camera framed for `at`, into `ring`'s
// slot; their addresses.
std::pair<MTL::GPUAddress, MTL::GPUAddress> Renderer::write_constants(
    FrameArray& ring, const FrameSlot& begun, const frame::FrameInputs& inputs,
    frame::Extent at, std::uint32_t accumulated_frames,
    std::optional<sampling::CameraJitter> jitter) {
  const contracts::FrameConstants constants =
      frame::frame_constants(inputs, at, accumulated_frames, jitter);
  const std::span<std::byte> bytes = ring.bytes(begun.slot);
  const MTL::GPUAddress address = ring.address(begun.slot);
  std::memcpy(bytes.subspan(constants_offset, sizeof(constants)).data(),
              &constants, sizeof(constants));
  MTL::GPUAddress camera_address = 0;
  if (inputs.camera) {
    const contracts::CameraData camera =
        camera::shader_form(*inputs.camera, at);
    std::memcpy(bytes.subspan(camera_offset, sizeof(camera)).data(), &camera,
                sizeof(camera));
    camera_address = address + camera_offset;
  }
  return {address + constants_offset, camera_address};
}

// Every pass's resources at the light size: the constants and camera, the
// scene, the images, and each family's (renderer.h). `previous` whether a
// temporal graph merges last frame.
FrameResources Renderer::frame_resources(const FrameSlot& begun,
                                         const frame::FrameInputs& inputs,
                                         MTL::Texture* target,
                                         frame::Extent light,
                                         const Prepared& prepared,
                                         bool previous) {
  const auto [constants_address, camera_address] = write_constants(
      constants_, begun, inputs, light, prepared.accumulated_frames,
      frame::camera_jitter(schedule_, inputs.index));
  FrameResources resources{
      .arguments = arguments_.get(),
      .constants = constants_address,
      .target = target,
      .size = light,
  };
  resources.camera = camera_address;
  if (scene_) {
    resources.scene = scene_->block_address();
    resources.transforms = transforms_->address(begun.slot);
    resources.glows = glows_->address(begun.slot);
    resources.acceleration = acceleration_->address(begun.slot);
    resources.mesh_block = acceleration_->mesh_block();
  }
  if (images_) {
    resources.radiance = images_->radiance();
    for (std::uint32_t level = 0; level < passes::bloom_levels; ++level) {
      resources.bloom.at(level) = images_->bloom(level);
    }
  }
  if (accumulation_) {
    // The accumulated image is the frame's radiance (frame_images.h).
    resources.radiance = accumulation_->texture();
    resources.accumulation = accumulation_->texture();
    resources.accumulated_frames = prepared.accumulated_frames;
    non_finite_->begin_frame(begun.slot, begun.sequence);
    resources.non_finite_counter = non_finite_->address(begun.slot);
  }
  if (checkerboard_ && !denoise_) {
    // The fill's light is what such a graph shows (renderer.h).
    resources.radiance = checkerboard_->images.filled();
  }
  const std::optional<contracts::Camera> previous_camera =
      previous ? previous_camera_ : std::nullopt;
  if (mnee_) {
    mnee_->bind(resources, begun);
  }
  if (restir_pt_) {
    restir_pt_->bind(resources, begun, inputs.index, previous_camera, light);
    // The grid its one-thread-a-pixel kernels run over (contract 20).
    resources.traced = frame::traced_grid(light, checkerboard_.has_value());
  }
  if (features_) {
    features_->bind(resources, begun, inputs.previous, light);
  }
  if (reuses_history_) {
    // Last frame's transforms and glows: this frame's own where the
    // scene is still, else animated by animate() at the previous time.
    resources.last_transforms = last_transforms_
                                    ? last_transforms_->address(begun.slot)
                                    : resources.transforms;
    resources.last_glows =
        last_glows_ ? last_glows_->address(begun.slot) : resources.glows;
  }
  return resources;
}

// Animate: every moving shape placed and every glowing light lit at the
// frame's time, by the core, in the slot's transforms and glows; and, when
// shapes move, the slot's structure rebuilt over them, before any pass
// traces it (renderer.h). With them, last frame's and the features' frame
// before's where the graph keeps them.
void Renderer::animate(const FrameSlot& begun, const frame::FrameInputs& inputs,
                       bool previous) {
  if (!scene_ || !animation::changes(animation_)) {
    return;
  }
  animation::animate(animation_, inputs.time,
                     transforms_->transforms(begun.slot),
                     glows_->glows(begun.slot));
  // Last frame's, at its time; unread at a history's first frame.
  if (last_transforms_ && previous) {
    animation::animate(animation_, previous_time_,
                       last_transforms_->transforms(begun.slot),
                       last_glows_->glows(begun.slot));
  }
  // The features' frame before, at its time; unread without one. Its
  // shapes alone: a glow does not move a surface.
  if (features_ && features_->previous_transforms && inputs.previous) {
    animation::place(animation_, inputs.previous->time,
                     features_->previous_transforms->transforms(begun.slot));
  }
}

void Renderer::update_structure(MTL4::ComputeCommandEncoder* encoder,
                                const FrameSlot& begun) {
  if (scene_ && animation::moves(animation_)) {
    acceleration_->update(encoder, begun.slot,
                          transforms_->transforms(begun.slot));
  }
}

// A denoised graph's passes from the denoise pass on (renderer.h): at the
// target's size, with its constants and camera, the radiance image the
// denoise pass writes and the presenting pass reads; the denoise pass's own
// inputs; and the full-size albedo's features dispatch. Precondition: the
// graph denoises.
Renderer::DenoiseResources Renderer::denoise_resources(
    const FrameResources& resources, const FrameSlot& begun,
    const frame::FrameInputs& inputs, const contracts::Camera& lens,
    frame::Extent size, const Prepared& prepared) {
  DenoiseFamily& d = made(denoise_, "denoiser");
  const frame::Extent light = resources.size;
  DenoiseResources out{resources, resources, resources, resources, false};
  FrameResources& output = out.output;
  const auto [output_constants, output_camera] =
      write_constants(d.output_constants, begun, inputs, size,
                      prepared.accumulated_frames, std::nullopt);
  output.constants = output_constants;
  output.camera = output_camera;
  output.size = size;
  output.radiance = images_->radiance();
  output.accumulation = nullptr;

  FrameResources& full_features = out.full_features;
  full_features = output;
  full_features.feature_constants = write_feature_constants(
      d.full_feature_constants, begun.slot, inputs.previous, size);
  const FeatureImages& full =
      made(d.full_features.at(begun.slot),
           "full-size features");  // the slot's (renderer.h)
  full_features.albedo = full.albedo();
  full_features.normal = full.normal();
  full_features.depth = full.depth();
  full_features.motion = full.motion();
  full_features.visibility = full.visibility();
  full_features.raster_depth = full.raster_depth();

  // Its constants: the view axes, the sizes, whether there is a history,
  // and the light's jitter, the frame's (renderer.h, "The camera's
  // jitter"): a denoised graph feeds the network, so its frames take it.
  const contracts::DenoiseConstants k = frame::denoise_constants(
      made(inputs.camera, "camera"),
      frame::DenoiseFrame{
          .light = light,
          .padded = d.images.padded(),
          .output = size,
          .features = d.features,
          .history = prepared.denoise_history,
          .jitter = made(frame::camera_jitter(schedule_, inputs.index),
                         "camera jitter"),
      });
  std::memcpy(d.constants.bytes(begun.slot).data(), &k, sizeof(k));

  FrameResources& denoising = out.denoising;
  denoising = output;
  // the hand-off (renderer.h)
  denoising.light_radiance = d.images.light_radiance(begun.slot);
  denoising.albedo = resources.albedo;
  denoising.normal = resources.normal;
  denoising.depth = resources.depth;
  denoising.motion = resources.motion;
  denoising.full_albedo = full.albedo();
  denoising.denoise_constants = d.constants.address(begun.slot);
  denoising.denoise = &d.images;

  // The depth of field's: at the output's size, the radiance image the
  // denoise pass wrote, the slot's full-size depth, its constants from the
  // frame's own lens (renderer.h, "The depth of field").
  DepthOfFieldFamily& f = made(depth_of_field_, "depth of field");
  const contracts::DepthOfFieldConstants dof =
      passes::depth_of_field_constants(lens, size);
  std::memcpy(f.constants.bytes(begun.slot).data(), &dof, sizeof(dof));
  FrameResources& blurred = out.depth_of_field;
  blurred = output;
  blurred.full_depth = full.depth();
  blurred.depth_of_field_constants = f.constants.address(begun.slot);
  blurred.depth_of_field = &f.images;
  out.blurs = passes::blurs(dof);
  return out;
}

// The light half's passes in order into `encoder`, each after the barriers
// its reads need (renderer.h): every pass before the denoise pass.
void Renderer::record_light(MTL4::ComputeCommandEncoder* encoder,
                            const FrameResources& resources, std::size_t first,
                            std::size_t last) {
  if (first == 0) {
    encoder->setArgumentTable(resources.arguments);
  }
  // whether a ReSTIR PT pass came before, in this frame
  bool restir_before = false;
  for (std::size_t i = first; i < last; ++i) {
    const Step& step = steps_[i];
    if (i == first_cross_frame_) {
      // The frames in flight share the images between passes, the
      // accumulated image and ReSTIR PT's arrays: none writes them while
      // the frame before still reads them, and none reads the
      // accumulated image before the frame before has written it. One
      // barrier for every such hazard, before the first pass that
      // touches any (GPU.8).
      encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch,
                                       MTL4::VisibilityOptionDevice);
    }
    if (std::holds_alternative<ShadedInMerge>(step.pass)) {
      continue;  // shaded by the spatial pass just before (renderer.h)
    }
    if (frame::is_restir_pt(step.kind) && restir_before) {
      // What the ReSTIR PT pass before it wrote this frame: the
      // reservoirs, primary hits and light buffer (GPU.8: a write and
      // its reader).
      barrier_after_dispatch(encoder);
    }
    restir_before = restir_before || frame::is_restir_pt(step.kind);
    if (frame::reads_radiance(step.kind)) {
      // The radiance image an earlier pass of this frame wrote.
      barrier_after_dispatch(encoder);
    }
    std::visit([&](const auto& pass) { record_pass(pass, encoder, resources); },
               step.pass);
  }
}

// The checkerboard's fill, writing `hand_off` whole from the light's
// radiance, after the shade's, the features' and, denoised, the full-size
// albedo's dispatches (renderer.h, "Two halves", "Checkerboard rendering";
// GPU.8: dispatch before dispatch): a denoised graph's slot's hand-off, or
// another's filled image. Precondition: the graph feeds the network.
void Renderer::record_hand_off(MTL4::ComputeCommandEncoder* encoder,
                               const FrameSlot& begun,
                               const FrameResources& resources,
                               const Prepared& prepared,
                               MTL::Texture* hand_off) {
  CheckerboardFamily& c = made(checkerboard_, "checkerboard fill");
  // The frame before's history is this history's frame before when the
  // fill's own history holds it (renderer.h).
  const contracts::CheckerboardConstants constants =
      frame::checkerboard_constants(prepared.fill_history);
  std::memcpy(c.constants.bytes(begun.slot).data(), &constants,
              sizeof(constants));
  // At the light size, as the light half's passes: its constants, the
  // accumulated image and this slot's features; this slot's hand-off; the
  // fill's history by the frame's parity, the frame before's and its own.
  FrameResources fill = resources;
  fill.checkerboard_constants = c.constants.address(begun.slot);
  fill.light_radiance = hand_off;
  fill.previous_light = c.images.before(prepared.index);
  fill.history_light = c.images.written(prepared.index);
  barrier_after_dispatch(encoder);
  c.fill.record(encoder, fill);
}

// The image half's passes in order into `encoder`, each after the barriers
// its reads need (renderer.h): the denoise pass and every pass after it.
bool Renderer::record_raster(const FrameSlot& begun,
                             const FrameResources& resources,
                             const FrameResources* full) const {
  bool recorded = false;
  for (const Step& step : steps_) {
    if (step.kind == frame::PassKind::features) {
      // A set shared by the frames in flight waits for their reads
      // (renderer.h: one set a slot where handed off).
      const bool shared = !made(features_, "features").sets[1].has_value();
      std::get<FeaturesPass>(step.pass).rasterize(begun.commands, resources,
                                                  shared);
      recorded = true;
    }
  }
  if (full != nullptr) {
    // one set a slot (renderer.h)
    made(denoise_, "denoiser")
        .full_features_pass.rasterize(begun.commands, *full, false);
    recorded = true;
  }
  return recorded;
}

void Renderer::record_image(MTL4::ComputeCommandEncoder*& encoder,
                            const DenoiseResources& denoised_resources) {
  for (std::size_t i = image_begins_; i < steps_.size(); ++i) {
    const Step& step = steps_[i];
    if (frame::reads_radiance(step.kind)) {
      // The radiance an earlier pass wrote: the light's, handed off,
      // for the denoise pass; the denoise pass's for the tone map.
      barrier_after_dispatch(encoder);
    }
    if (step.kind == frame::PassKind::denoise) {
      // Its network off the GPU, between the halves of the image part: it
      // leaves the part after's compute encoder in `encoder` for the passes
      // after it (denoise.h).
      std::get<DenoisePass>(step.pass).record(encoder,
                                              denoised_resources.denoising);
      continue;
    }
    if (step.kind == frame::PassKind::depth_of_field) {
      // Nothing for a pinhole's frame (renderer.h, "The depth of field").
      if (denoised_resources.blurs) {
        std::visit(
            [&](const auto& pass) {
              record_pass(pass, encoder, denoised_resources.depth_of_field);
            },
            step.pass);
      }
      continue;
    }
    std::visit(
        [&](const auto& pass) {
          record_pass(pass, encoder, denoised_resources.output);
        },
        step.pass);
  }
}

void Renderer::record(const FrameSlot& begun, const frame::FrameInputs& given,
                      MTL::Texture* target, frame::Extent size) {
  if (begun.commands == nullptr || begun.slot >= frames_in_flight ||
      target == nullptr || size.width == 0 || size.height == 0) {
    throw MetalError(
        "Renderer::record: no command buffer, no frame slot, no target, or an "
        "empty image");
  }
  if (begun.second != nullptr && parts() == Parts::one) {
    throw MetalError(
        "Renderer::record: a submission of two parts for a frame graph of one "
        "(Renderer::parts)");
  }
  refuse(frame::time_refusal(given.time, given.index, "its time"), "");
  const Prepared prepared = take_prepared(given, size);
  if (needs_scene_ && !given.camera) {
    throw MetalError(
        "Renderer::record: the frame graph reads a scene, and the frame has no "
        "camera");
  }
  // The frame as its light is traced: through a pinhole in a graph that
  // applies the lens after the network (renderer.h, "The depth of field"),
  // or whose features Metal rasterizes (raster_inputs.h).
  const frame::FrameInputs inputs = raster_inputs(schedule_, given);
  // The passes before the denoise pass render at the light size, the
  // target's own in a graph without it (renderer.h).
  const frame::Extent light = frame::light_size(schedule_, size);
  // A temporal graph merges last frame when the history holds it and the
  // renderer recorded it (its camera kept): never at a history's first.
  const bool previous = reuses_history_ && prepared.reused_frames > 0 &&
                        previous_camera_.has_value();
  FrameResources resources =
      frame_resources(begun, inputs, target, light, prepared, previous);
  // The frame's shapes and glows placed on the host; then what the
  // rasterizer draws of them this frame (scene_acceleration.h, raster()).
  animate(begun, inputs, previous);
  if (scene_ && inputs.camera) {
    const SceneAcceleration::Raster raster = acceleration_->raster(
        begun.slot, transforms_->transforms(begun.slot), *inputs.camera, light);
    resources.draws = raster.draws;
    resources.instance_shapes = raster.instances;
  }
  // A denoised graph's own, before the encoders: its full-size features'
  // raster pass opens the frame beside the light size's.
  std::optional<DenoiseResources> denoised;
  if (denoise_) {
    denoised.emplace(denoise_resources(resources, begun, inputs,
                                       made(given.camera, "camera"), size,
                                       prepared));
  }
  const bool rasterized = record_raster(
      begun, resources, denoised ? &denoised->full_features : nullptr);

  MTL4::ComputeCommandEncoder* encoder =
      begun.commands->computeCommandEncoder();
  if (encoder == nullptr) {
    throw MetalError(
        "Renderer::record: the command buffer made no compute encoder");
  }
  if (rasterized) {
    // The features read the visibility buffers the raster passes drew.
    encoder->barrierAfterQueueStages(MTL::StageFragment, MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
  }
  update_structure(encoder, begun);
  record_light(encoder, resources, 0, fill_at_);
  if (!denoise_) {
    if (checkerboard_) {
      // The fill, then the presenting pass (renderer.h, "Checkerboard
      // rendering").
      record_hand_off(encoder, begun, resources, prepared,
                      checkerboard_->images.filled());
      record_light(encoder, resources, fill_at_, steps_.size());
    }
    encoder->endEncoding();
  } else {
    // The full-size albedo, in the light half (renderer.h, "Two
    // halves"): it writes the slot's own images, which nothing before
    // it in the frame reads, so no barrier before it, as none before the
    // light-size features; the fill's barrier orders it before the image
    // half in one part, the event in two.
    DenoiseFamily& d = made(denoise_, "denoiser");
    const DenoiseResources& denoising =
        made(denoised, "the denoised frame's resources");
    d.full_features_pass.record(encoder, denoising.full_features);
    record_hand_off(encoder, begun, resources, prepared,
                    d.images.light_radiance(begun.slot));
    MTL4::ComputeCommandEncoder* image = encoder;
    if (begun.second != nullptr) {
      // The image half in the second part, on the overlap queue
      // (renderer.h, "Two halves"): after the light half by the
      // submission's event, and after the frame before's image half,
      // the overlap queue's earlier dispatches, by this barrier (GPU.8).
      encoder->endEncoding();
      image = begun.second->computeCommandEncoder();
      if (image == nullptr) {
        throw MetalError(
            "Renderer::record: the second command buffer made no compute "
            "encoder");
      }
      image->setArgumentTable(arguments_.get());
      image->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch,
                                     MTL4::VisibilityOptionDevice);
    } else {
      // One part: the fill before the image half reads its hand-off, in
      // place of the event (renderer.h, "One part for a denoised graph").
      barrier_after_dispatch(encoder);
    }
    record_image(image, denoising);
    image->endEncoding();
  }
  // What the next frame's temporal merge asks of this one; what
  // read_network_input() reads of it.
  previous_camera_ = inputs.camera;
  previous_time_ = inputs.time;
  last_slot_ = begun.slot;
  last_jitter_ = frame::camera_jitter(schedule_, inputs.index)
                     .value_or(sampling::CameraJitter{});
}

WindowFrame render_to_window(Submission& submission, const Presenter& presenter,
                             Renderer& renderer,
                             const frame::FrameInputs& inputs,
                             CA::MetalDrawable* drawable) {
  // The passes dispatch a thread per pixel of the presenter's size, and
  // write the drawable's texture (renderer.h; I.6).
  if (!presenter.renders_into(drawable)) {
    throw MetalError("render_to_window: the drawable is not the " +
                     frame::to_string(presenter.size()) +
                     " the presenter renders at");
  }
  const auto pool = scoped_pool();
  const frame::Extent size = presenter.size();
  renderer.prepare(inputs, size);
  const FrameSlot begun = submission.begin(renderer.parts());
  renderer.record(begun, inputs, drawable->texture(), size);
  submission.present(drawable);
  return WindowFrame{begun.sequence, begun.settled};
}

WindowFrame render_offscreen_frame(Submission& submission, Offscreen& target,
                                   Renderer& renderer,
                                   const frame::FrameInputs& inputs) {
  // What recording gets from Metal autoreleased (the frame's encoder) is
  // released at the end of the frame, not whenever the caller's pool, if it
  // has one, drains (P.8).
  const auto pool = scoped_pool();
  renderer.prepare(inputs, target.size());
  const FrameSlot begun = submission.begin(renderer.parts());
  renderer.record(begun, inputs, target.texture(), target.size());
  submission.commit();
  return WindowFrame{begun.sequence, begun.settled};
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
  return render_offscreen_frame(submission, target, renderer, inputs).sequence;
}

}  // namespace serenity::metal
