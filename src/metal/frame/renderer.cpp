#include "metal/frame/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/animation/animate.h"
#include "core/camera/thin_lens.h"
#include "core/contracts/camera.h"
#include "core/contracts/denoise.h"
#include "core/contracts/features.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/reuse_constants.h"
#include "core/contracts/surface_record.h"
#include "core/denoise/network.h"
#include "core/sampling/pairing.h"
#include "core/scene/packed_range.h"
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
static_assert(constants_offset + sizeof(contracts::FrameConstants) <= camera_offset);
static_assert(camera_offset + sizeof(contracts::CameraData) <= ring_slot_size);
static_assert(constants_offset % constant_offset_alignment == 0 && camera_offset % constant_offset_alignment == 0);
static_assert(camera_offset % alignof(contracts::CameraData) == 0);
// Both are written into the ring as their bytes (SL.con.4, COPY.6).
static_assert(std::is_trivially_copyable_v<contracts::FrameConstants>);
static_assert(std::is_trivially_copyable_v<contracts::CameraData>);

// The argument table's sizes (renderer.h). Each family of passes binds at
// most its constant's buffers and textures through the table
// (metal/passes/bindings.h), each the index past its passes' last, asserted
// below rather than restated in words; a graph's table holds the most of
// the families it has (buffer_bindings()). A new kind of material,
// texture, medium or light adds to the scene's block, not here.
//
// Every graph: the path pass without mnee, the preview, the features pass,
// the presenting passes.
constexpr NS::UInteger max_buffers = 8;
constexpr NS::UInteger max_textures = 4;
static_assert(bindings::path::non_finite < max_buffers);
static_assert(bindings::preview::transforms < max_buffers);
static_assert(bindings::features::last_transforms < max_buffers);
static_assert(bindings::tone_map::target < max_textures);
static_assert(bindings::features::motion < max_textures);
// A graph with mnee: the path pass's work list and the MNEE pass.
constexpr NS::UInteger max_mnee_buffers = 12;
static_assert(bindings::path::mnee_header < max_mnee_buffers);
static_assert(bindings::mnee::lost < max_mnee_buffers);
// A graph with the denoise pass.
constexpr NS::UInteger max_denoise_buffers = 12;
constexpr NS::UInteger max_denoise_textures = 8;
static_assert(bindings::denoise::coarse < max_denoise_buffers);
static_assert(bindings::denoise::radiance < max_denoise_textures);
// A ReSTIR DI graph.
constexpr NS::UInteger max_restir_buffers = 15;
static_assert(bindings::restir_initial::output < max_restir_buffers);
static_assert(bindings::restir_temporal::last_transforms < max_restir_buffers);
static_assert(bindings::restir_spatial::output < max_restir_buffers);
static_assert(bindings::restir_shade::non_finite < max_restir_buffers);
// A ReSTIR PT graph, with or without mnee (its initial pass's work list and
// the MNEE pass bind within it).
constexpr NS::UInteger max_restir_pt_buffers = 28;
static_assert(bindings::restir_pt::mnee_header < max_restir_pt_buffers);
static_assert(bindings::mnee::lost < max_restir_pt_buffers);

// The reuse textures' seed: fixed, so every run pairs the same pixels
// (core/sampling/pairing.h).
constexpr std::uint64_t pairing_seed = 0x5e2e417f;

// Whether `kind` is a pass of either ReSTIR, whose arrays the frames share.
bool reuses_samples(frame::PassKind kind) {
    return frame::is_restir(kind) || frame::is_restir_pt(kind);
}

// Whether `schedule` has a pass of `kind`.
bool has(const frame::Schedule& schedule, frame::PassKind kind) {
    return std::ranges::find(schedule.passes, kind) != schedule.passes.end();
}

// The reuse constants are written into their ring as their bytes.
static_assert(std::is_trivially_copyable_v<contracts::ReuseConstants>);

// The buffers `schedule`'s table holds: the most of its families' (above).
NS::UInteger buffer_bindings(const frame::Schedule& schedule) {
    NS::UInteger most = max_buffers;
    if (schedule.mnee) {
        most = std::max(most, max_mnee_buffers);
    }
    if (frame::denoises(schedule)) {
        most = std::max(most, max_denoise_buffers);
    }
    if (std::ranges::any_of(schedule.passes, frame::is_restir)) {
        most = std::max(most, max_restir_buffers);
    }
    if (std::ranges::any_of(schedule.passes, frame::is_restir_pt)) {
        most = std::max(most, max_restir_pt_buffers);
    }
    return most;
}

// The textures it holds.
NS::UInteger texture_bindings(const frame::Schedule& schedule) {
    return frame::denoises(schedule) ? max_denoise_textures : max_textures;
}

// What the path state keeps of a medium's index: 16 bits, 0xffff for none
// (core/contracts/surface_record.h). A ReSTIR graph refuses a scene with
// that many media or more (I.5).
constexpr std::size_t max_restir_media = contracts::path_no_medium;

// Shaders read a time as a float (contracts/frame_constants.h): a double
// past float's range would reach them as infinity. `which` names the time
// for the error: frame `index`'s own, or the frame before's.
void check_time(frame::Seconds time, std::uint64_t index, std::string_view which) {
    const double seconds = time.count();
    if (!std::isfinite(seconds) || std::abs(seconds) > std::numeric_limits<float>::max()) {
        throw MetalError("frame " + std::to_string(index) + ": " + std::string(which) + ", " +
                         std::to_string(seconds) + " s, is past what the shaders' float holds");
    }
}

// What a reuse family's G-buffer or path state packs of `scene`
// (contracts/surface_record.h, primary_hit.h; core/scene/packed_range.h),
// refused rather than wrapped (I.5): fewer media than `media_limit`, which
// a medium's 16-bit index holds with its "none", and every material value
// within the packed BSDF's range. `family` names the reuse in the error.
void check_packable(const scene::SceneDescription& scene, std::size_t media_limit, std::string_view family) {
    if (scene.media.size() >= media_limit) {
        throw MetalError("Renderer: the scene has " + std::to_string(scene.media.size()) + " media, and " +
                         std::string(family) + " keeps a medium's index in 16 bits, under " +
                         std::to_string(media_limit));
    }
    if (const std::optional<std::string> outside = scene::outside_packed_bsdf(scene)) {
        throw MetalError("Renderer: the scene's " + *outside);
    }
}

NS::SharedPtr<MTL4::ArgumentTable> make_argument_table(MTL::Device* device, NS::UInteger buffers,
                                                      NS::UInteger textures) {
    const auto pool = scoped_pool();  // Metal's error, if any, is autoreleased
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(buffers);
    descriptor->setMaxTextureBindCount(textures);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(device->newArgumentTable(descriptor.get(), &error));
    if (!table) {
        throw MetalError("Renderer: the device made no argument table: " + describe(error));
    }
    return table;
}

// The features pass's constants for a frame of `size` (contract 17) into
// `ring`'s `slot`: whether there is a frame before, and its camera framed
// for `size`. Their address. For the features at the light size and the
// denoiser's full-size albedo alike.
MTL::GPUAddress write_feature_constants(FrameArray& ring, std::uint32_t slot,
                                        const std::optional<frame::PreviousFrame>& previous, frame::Extent size) {
    contracts::FeatureConstants features{};
    features.previous = previous ? 1u : 0u;
    if (previous) {
        features.previous_camera = camera::shader_form(previous->camera, size);
    }
    std::memcpy(ring.bytes(slot).data(), &features, sizeof(features));
    return ring.address(slot);
}

}  // namespace

// ReSTIR DI's (renderer.h): its settings, its arrays, and the ring of its
// reuse constants.
Renderer::RestirDiFamily::RestirDiFamily(const Device& device, Submission& submission,
                                         const passes::RestirDi& graph_settings, bool temporal, bool spatial)
    : settings(graph_settings),
      images(device, submission, temporal ? ReuseImages::Temporal::merged : ReuseImages::Temporal::none,
             temporal || spatial ? ReuseImages::Working::needed : ReuseImages::Working::none),
      constants(device, submission, std::vector<std::byte>(sizeof(contracts::ReuseConstants)),
                FrameArray::Copies::per_frame) {}

void Renderer::RestirDiFamily::bind(FrameResources& resources, const FrameSlot& begun, std::uint64_t index,
                                    const std::optional<contracts::Camera>& previous_camera, frame::Extent light) {
    // The reuse constants: the settings, and whether there is a frame
    // before to merge, with its camera framed for this frame's size.
    contracts::ReuseConstants reuse{};
    reuse.settings = settings;
    reuse.previous = previous_camera ? 1u : 0u;
    if (previous_camera) {
        reuse.previous_camera = camera::shader_form(*previous_camera, light);
    }
    std::memcpy(constants.bytes(begun.slot).data(), &reuse, sizeof(reuse));
    resources.reuse = constants.address(begun.slot);
    const ReuseImages::Addresses arrays = images.addresses(index);
    resources.records = arrays.records;
    resources.path_states = arrays.path_states;
    resources.previous_positions = arrays.previous_positions;
    resources.last_records = arrays.last_records;
    resources.working_reservoirs = arrays.working;
    resources.final_reservoirs = arrays.final_reservoirs;
    resources.last_final_reservoirs = arrays.last_final;
}

// ReSTIR PT's (renderer.h): its arrays, its settings and reuse textures,
// and with temporal reuse the ring of its reuse constants.
Renderer::RestirPtFamily::RestirPtFamily(const Device& device, Submission& submission,
                                         const contracts::PathReuseSettings& settings,
                                         std::span<const std::uint16_t> textures, bool spatial, bool temporal)
    : images(device, submission, spatial ? PathReuseImages::Spatial::merged : PathReuseImages::Spatial::none,
             temporal ? PathReuseImages::Temporal::merged : PathReuseImages::Temporal::none),
      statics(device, submission,
              std::array<std::span<const std::byte>, 2>{std::as_bytes(std::span(&settings, 1)),
                                                        std::as_bytes(textures)}) {
    if (temporal) {
        constants.emplace(device, submission, std::vector<std::byte>(sizeof(contracts::PathReuseConstants)),
                          FrameArray::Copies::per_frame);
    }
}

void Renderer::RestirPtFamily::bind(FrameResources& resources, const FrameSlot& begun, std::uint64_t index,
                                    const std::optional<contracts::Camera>& previous_camera, frame::Extent light) {
    const PathReuseImages::Addresses arrays = images.addresses(index);
    resources.path_settings = statics.address(0);
    resources.pairing = statics.address(1);
    resources.path_reservoirs = arrays.reservoirs;
    resources.primaries = arrays.primaries;
    resources.light = arrays.light;
    resources.shifts = arrays.shifts;
    resources.replays = arrays.replays;
    resources.replay_counter = arrays.counter;
    resources.replay_grid = arrays.grid;
    resources.light_tiles = arrays.light_tiles;
    resources.last_primaries = arrays.last_primaries;
    resources.path_finals = arrays.finals;
    resources.last_path_finals = arrays.last_finals;
    resources.previous_positions = arrays.previous_positions;
    resources.previous_pixels = arrays.previous_pixels;
    resources.duplication = arrays.duplication;
    if (constants) {
        // Whether there is a frame before to merge, with its camera framed
        // for this frame's size.
        contracts::PathReuseConstants reuse{};
        reuse.previous = previous_camera ? 1u : 0u;
        if (previous_camera) {
            reuse.previous_camera = camera::shader_form(*previous_camera, light);
        }
        std::memcpy(constants->bytes(begun.slot).data(), &reuse, sizeof(reuse));
        resources.path_constants = constants->address(begun.slot);
    }
}

// MNEE's (renderer.h): its work list and its count of lost requests.
Renderer::MneeFamily::MneeFamily(const Device& device, Submission& submission, MneeList::Light light)
    : list(device, submission, light), lost(device, submission, "MNEE requests lost to a full work list") {}

void Renderer::MneeFamily::bind(FrameResources& resources, const FrameSlot& begun) {
    const MneeList::Addresses addresses = list.addresses();
    resources.mnee_requests = addresses.requests;
    resources.mnee_header = addresses.header;
    resources.mnee_grid = addresses.grid;
    resources.mnee_light = addresses.light;
    lost.begin_frame(begun.slot, begun.sequence);
    resources.mnee_lost_counter = lost.address(begun.slot);
}

// The features pass's (renderer.h): its images, its constants' ring and,
// when shapes move, the shapes as the frame before placed them.
Renderer::FeatureFamily::FeatureFamily(const Device& device, Submission& submission,
                                       const scene::SceneDescription& scene, bool moves)
    : images(device, submission),
      constants(device, submission, std::vector<std::byte>(sizeof(contracts::FeatureConstants)),
                FrameArray::Copies::per_frame) {
    if (moves) {
        previous_transforms.emplace(device, submission, scene.shapes.transforms, FrameArray::Copies::per_frame);
    }
}

void Renderer::FeatureFamily::bind(FrameResources& resources, const FrameSlot& begun,
                                   const std::optional<frame::PreviousFrame>& previous, frame::Extent light) {
    // Whether there is a frame before, and its camera framed for this
    // frame's size; its shapes, animated in record() where they move.
    resources.feature_constants = write_feature_constants(constants, begun.slot, previous, light);
    resources.feature_last_transforms =
        previous_transforms ? previous_transforms->address(begun.slot) : resources.transforms;
    resources.albedo = images.albedo();
    resources.normal = images.normal();
    resources.depth = images.depth();
    resources.motion = images.motion();
}

// The denoise pass's (renderer.h): the network's last width, its buffers,
// the target-size ring, its constants' ring, and the full-size albedo's
// images, constants and kernel.
Renderer::DenoiseFamily::DenoiseFamily(const Device& device, Submission& submission, const Library& library,
                                       const denoise::Network& network)
    : features(network.widths.dec.back()),
      images(device, submission, network.widths),
      output_constants(device, submission, std::vector<std::byte>(ring_slot_size), FrameArray::Copies::per_frame),
      constants(device, submission, std::vector<std::byte>(sizeof(contracts::DenoiseConstants)),
                FrameArray::Copies::per_frame),
      full_features(device, submission),
      full_feature_constants(device, submission, std::vector<std::byte>(sizeof(contracts::FeatureConstants)),
                             FrameArray::Copies::per_frame),
      full_features_pass(library) {}

Renderer::Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : submission_(submission),
      library_(device, serenity::metallib::shaders),
      constants_(device, submission, std::vector<std::byte>(ring_slot_size), FrameArray::Copies::per_frame),
      arguments_(make_argument_table(device.handle(), buffer_bindings(schedule), texture_bindings(schedule))),
      denoises_(frame::denoises(schedule)),
      schedule_(schedule) {
    // The core decides which schedules can be carried out (principle 10).
    if (const std::optional<std::string> reason = frame::invalid(schedule)) {
        throw MetalError("Renderer: " + *reason);
    }
    needs_scene_ = std::ranges::any_of(schedule.passes, frame::needs_scene);
    if (needs_scene_ && scene == nullptr) {
        throw MetalError("Renderer: the frame graph's passes read a scene, and none was given (--scene)");
    }
    const auto pool = scoped_pool();
    // Raytraced mode's temporal merge traces last frame's surface against
    // last frame's structure, which this backend does not keep: refused by
    // name, never traced against this frame's (renderer.h; the design for
    // keeping it waits in docs/process/QUEUE.md).
    if (has(schedule, frame::PassKind::restir_temporal) && schedule.restir_di &&
        schedule.restir_di->bias_correction == contracts::BiasCorrection::raytraced) {
        throw MetalError("Renderer: the frame graph's restir_temporal pass with bias_correction = \"raytraced\" is "
                         "not implemented by the Metal backend yet: it needs last frame's acceleration structure; "
                         "use \"basic\"");
    }
    // The network, read once (core/denoise/network.h); its weights go to the
    // GPU with its pass, its widths size its buffers.
    std::optional<denoise::Network> network;
    if (schedule.denoise) {
        network = denoise::load_network(schedule.denoise->weights);
    }
    make_steps(device, schedule, network);
    make_images(device, schedule);
    make_restir_di(device, schedule, scene);
    make_restir_pt(device, schedule, scene);
    make_mnee(device, schedule, scene);
    if (needs_scene_) {
        make_scene(device, *scene);
    }
    // The features pass's. Its pass reads a scene, which the check above
    // has made sure of.
    if (frame::writes_features(schedule)) {
        features_.emplace(device, submission_, *scene, animation::moves(animation_));
    }
    // The denoise pass's. The core has made sure the graph has the features
    // pass and a light pass before it (frame::invalid).
    if (denoises_) {
        denoise_.emplace(device, submission_, library_, *network);
    }
}

// Each of the schedule's pass kinds as this backend's pass for it, in
// schedule order (renderer.h); and the first pass that writes the images
// between passes, accumulates, or writes ReSTIR's arrays, which the frame
// before's shade reads, or the feature images, which the frame before's
// readers read.
void Renderer::make_steps(const Device& device, const frame::Schedule& schedule,
                          const std::optional<denoise::Network>& network) {
    const auto crosses_frames = [](frame::PassKind kind) {
        return frame::writes_radiance(kind) || frame::accumulates(kind) || reuses_samples(kind) ||
               frame::writes_features(kind) || kind == frame::PassKind::denoise;
    };
    if (const auto first = std::ranges::find_if(schedule.passes, crosses_frames); first != schedule.passes.end()) {
        first_cross_frame_ = static_cast<std::size_t>(first - schedule.passes.begin());
    }
    const bool temporal = has(schedule, frame::PassKind::restir_temporal);
    const bool spatial = has(schedule, frame::PassKind::restir_spatial);
    steps_.reserve(schedule.passes.size());
    for (frame::PassKind kind : schedule.passes) {
        // No default: a kind this backend does not implement fails the build
        // (core/frame/schedule.h).
        switch (kind) {
        case frame::PassKind::test_pattern:
            steps_.push_back(Step{kind, Pass{std::in_place_type<TestPatternPass>, library_}});
            break;
        case frame::PassKind::preview:
            steps_.push_back(Step{kind, Pass{std::in_place_type<PreviewPass>, library_}});
            break;
        case frame::PassKind::path:
            steps_.push_back(Step{kind, Pass{std::in_place_type<PathPass>, library_, schedule.mnee}});
            break;
        case frame::PassKind::display:
            steps_.push_back(Step{kind, Pass{std::in_place_type<DisplayPass>, library_}});
            break;
        case frame::PassKind::tone_map:
            // The schedule has its settings with its pass (frame::invalid).
            steps_.push_back(
                Step{kind, Pass{std::in_place_type<ToneMapPass>, device, library_, submission_, *schedule.tone_map}});
            break;
        case frame::PassKind::restir_initial:
            // Into the working array when a merge follows, else the frame's
            // final one (frame_resources.h).
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirInitialPass>, library_,
                                             temporal || spatial ? ReservoirOutput::working : ReservoirOutput::final}});
            break;
        case frame::PassKind::restir_temporal:
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirTemporalPass>, library_,
                                             spatial ? ReservoirOutput::working : ReservoirOutput::final}});
            break;
        case frame::PassKind::restir_spatial:
            // Always to the final array, from the working one.
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirSpatialPass>, library_}});
            break;
        case frame::PassKind::restir_shade:
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirShadePass>, library_}});
            break;
        case frame::PassKind::restir_pt_initial:
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirPtInitialPass>, library_, schedule.mnee}});
            break;
        case frame::PassKind::restir_pt_spatial:
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirPtSpatialPass>, library_}});
            break;
        case frame::PassKind::restir_pt_shade:
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirPtShadePass>, library_}});
            break;
        case frame::PassKind::restir_pt_temporal:
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirPtTemporalPass>, library_}});
            break;
        case frame::PassKind::restir_pt_duplication:
            steps_.push_back(Step{kind, Pass{std::in_place_type<RestirPtDuplicationPass>, library_}});
            break;
        case frame::PassKind::features:
            steps_.push_back(Step{kind, Pass{std::in_place_type<FeaturesPass>, library_}});
            break;
        case frame::PassKind::denoise:
            // The schedule has its settings with its pass (frame::invalid).
            steps_.push_back(
                Step{kind, Pass{std::in_place_type<DenoisePass>, device, library_, submission_, *network}});
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
void Renderer::make_images(const Device& device, const frame::Schedule& schedule) {
    const bool accumulates = frame::accumulates(schedule);
    const bool radiance = std::ranges::any_of(schedule.passes, frame::writes_radiance);
    const FrameImages::Radiance own =
        accumulates && !denoises_ ? FrameImages::Radiance::accumulated : FrameImages::Radiance::image;
    const FrameImages::Bloom bloom = schedule.tone_map ? FrameImages::Bloom::pyramid : FrameImages::Bloom::none;
    if (radiance && (own == FrameImages::Radiance::image || bloom == FrameImages::Bloom::pyramid)) {
        images_ = std::make_unique<FrameImages>(device, submission_, own, bloom);
    }
    if (accumulates) {
        accumulation_ = std::make_unique<Accumulation>(device, submission_);
        non_finite_ = std::make_unique<FailureCount>(device, submission_, "samples not finite");
    }
}

// ReSTIR DI's, only for its graph (renderer.h). Its passes read a scene,
// which the constructor has made sure of.
void Renderer::make_restir_di(const Device& device, const frame::Schedule& schedule,
                              const scene::SceneDescription* scene) {
    if (!schedule.restir_di) {
        return;
    }
    check_packable(*scene, max_restir_media, "ReSTIR DI's path state");
    const bool temporal = has(schedule, frame::PassKind::restir_temporal);
    reuses_history_ = temporal;
    restir_di_.emplace(device, submission_, *schedule.restir_di, temporal,
                       has(schedule, frame::PassKind::restir_spatial));
}

// ReSTIR PT's, only for its graph (renderer.h).
void Renderer::make_restir_pt(const Device& device, const frame::Schedule& schedule,
                              const scene::SceneDescription* scene) {
    if (!schedule.restir_pt) {
        return;
    }
    // What the G-buffer and the light tiles pack (contracts/primary_hit.h,
    // light_tile.h), refused rather than wrapped (I.5).
    check_packable(*scene, contracts::primary_no_medium, "ReSTIR PT's G-buffer");
    if (scene->shapes.records.size() > contracts::tile_largest_id + 1u ||
        scene->light_counts.lights > contracts::tile_largest_id + 1u) {
        throw MetalError("Renderer: the scene has " + std::to_string(scene->shapes.records.size()) + " shapes and " +
                         std::to_string(scene->light_counts.lights) +
                         " lights, and ReSTIR PT's light tiles keep each index in 16 bits, at most " +
                         std::to_string(contracts::tile_largest_id + 1u) + " of either");
    }
    const bool temporal = has(schedule, frame::PassKind::restir_pt_temporal);
    reuses_history_ = temporal;
    const bool spatial = has(schedule, frame::PassKind::restir_pt_spatial);
    contracts::PathReuseSettings settings = *schedule.restir_pt;
    settings.mnee = schedule.mnee ? 1u : 0u;  // the graph's, for every pass of the family
    // With spatial reuse, the three reuse textures, built once on the CPU
    // from the graph's sigma and a fixed seed (core/sampling/pairing.h); one
    // zero texel without.
    const std::vector<std::uint16_t> textures =
        spatial ? sampling::reuse_textures(static_cast<double>(settings.sigma), pairing_seed)
                : std::vector<std::uint16_t>(1, 0u);
    restir_pt_.emplace(device, submission_, settings, textures, spatial, temporal);
}

// MNEE's work list, with mnee; the path pass's own light buffer before its
// fold (ReSTIR PT's requests add to its light buffer). Its requests keep a
// shape in 16 bits (contract 16), refused rather than wrapped (I.5).
void Renderer::make_mnee(const Device& device, const frame::Schedule& schedule, const scene::SceneDescription* scene) {
    if (!schedule.mnee) {
        return;
    }
    if (scene->shapes.records.size() > contracts::mnee_largest_shape + 1u) {
        throw MetalError("Renderer: the scene has " + std::to_string(scene->shapes.records.size()) +
                         " shapes, and MNEE's requests keep a shape's index in 16 bits, at most " +
                         std::to_string(contracts::mnee_largest_shape + 1u));
    }
    mnee_.emplace(device, submission_, schedule.restir_pt ? MneeList::Light::none : MneeList::Light::own);
}

// The scene on the GPU, for a graph that reads it, and what its frames
// place (renderer.h).
void Renderer::make_scene(const Device& device, const scene::SceneDescription& scene) {
    scene_ = std::make_unique<SceneBuffers>(device, submission_, scene);
    animation_ = scene.animation;
    const bool moves = animation::moves(animation_);
    transforms_ = std::make_unique<ShapeTransforms>(device, submission_, scene.shapes.transforms,
                                                    moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
    // Checked here, not by animate() in a frame, so nothing about the scene
    // can fail once a frame's submission has begun (renderer.h).
    for (const animation::Glower& glower : animation_.glowers) {
        if (glower.target >= scene.light_counts.spheres) {
            throw MetalError("Renderer: a glowing light's target, " + std::to_string(glower.target) +
                             ", is not one of the scene's " + std::to_string(scene.light_counts.spheres) +
                             " sphere lights");
        }
    }
    glows_ = std::make_unique<LightGlows>(device, submission_, scene.light_counts.spheres,
                                          !animation_.glowers.empty());
    // Last frame's, for a temporal graph whose scene changes (renderer.h).
    if (reuses_history_ && animation::changes(animation_)) {
        last_transforms_ = std::make_unique<ShapeTransforms>(
            device, submission_, scene.shapes.transforms,
            moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
        last_glows_ = std::make_unique<LightGlows>(device, submission_, scene.light_counts.spheres,
                                                   !animation_.glowers.empty());
    }
    // Where the families meet: the acceleration structure learns which
    // shapes move, by the movers' targets, and nothing else of them.
    std::vector<std::uint32_t> moving;
    moving.reserve(animation_.movers.size());
    for (const animation::Mover& mover : animation_.movers) {
        moving.push_back(mover.target);
    }
    acceleration_ = std::make_unique<SceneAcceleration>(device, submission_, scene.shapes, moving);
}

Renderer::~Renderer() {
    // The pipelines and the argument table are not in the residency set, so
    // no Resident waits for them: this does, before any member is released.
    submission_.wait_idle();
}

void Renderer::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
    check_time(inputs.time, inputs.index, "its time");
    if (needs_scene_ && !inputs.camera) {
        throw MetalError("Renderer: frame " + std::to_string(inputs.index) +
                         ": the frame graph reads a scene, and the frame has no camera");
    }
    // The frame before, which the features' motion is measured from: its
    // camera framed and its time placing the shapes in record(), so both are
    // checked here, before the submission begins.
    if (features_ && inputs.previous) {
        check_time(inputs.previous->time, inputs.index, "the frame before's time");
        if (const std::optional<std::string_view> reason = camera::invalid(inputs.previous->camera)) {
            throw MetalError("Renderer: frame " + std::to_string(inputs.index) +
                             ": the frame before's camera cannot be framed: " + std::string(*reason));
        }
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
        features_->images.prepare(light);
    }
    if (denoise_) {
        denoise_->full_features.prepare(size);
    }
    if (mnee_) {
        mnee_->list.prepare(light);
    }
    // 0 without a temporal merge, of either family (never both in a graph).
    std::uint64_t reused = 0;
    if (restir_di_) {
        reused = restir_di_->images.prepare(inputs, light);
    } else if (restir_pt_) {
        reused = restir_pt_->images.prepare(inputs, light);
    }
    std::uint32_t held = 0;
    if (accumulation_) {
        const frame::SceneMotion motion =
            animation::changes(animation_) ? frame::SceneMotion::changing : frame::SceneMotion::still;
        held = accumulation_->prepare(inputs, light, motion);
    }
    const bool denoise_history = denoise_ ? denoise_->images.prepare(inputs, light) : false;
    prepared_ = Prepared{inputs.index, size, held, reused, denoise_history};
}

bool Renderer::prepares() const noexcept {
    return accumulation_ || images_ || restir_di_ || restir_pt_ || mnee_ || features_ || denoise_;
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
        image = images_ ? images_->radiance() : nullptr;  // the denoised radiance, at the target's size
    } else if (accumulation_) {
        image = accumulation_->texture();
    }
    if (image == nullptr) {
        throw MetalError("read_accumulated: the frame graph accumulates nothing, or no frame has made its image");
    }
    constexpr std::size_t channels = contracts::accumulated_image_channels;  // RGBA32Float
    const std::size_t row_bytes = std::size_t{image->width()} * channels * sizeof(float);
    const std::size_t floats = std::size_t{image->width()} * image->height() * channels;
    if (out.size() != floats) {
        throw MetalError("read_accumulated: " + std::to_string(out.size()) + " floats given for an image of " +
                         std::to_string(floats));
    }
    const frame::Extent size{static_cast<std::uint32_t>(image->width()), static_cast<std::uint32_t>(image->height())};
    const std::array<ImageCopy, 1> copy = {ImageCopy{image, 0, row_bytes}};
    const auto buffer =
        read_back(image->device(), submission_, size, row_bytes * size.height, copy, "read_accumulated");
    std::memcpy(out.data(), buffer->contents(), floats * sizeof(float));
}

FeatureImages::Read Renderer::read_features() const {
    if (!features_ || features_->images.albedo() == nullptr) {
        throw MetalError("read_features: the frame graph has no features pass, or no frame has made its images");
    }
    return features_->images.read();
}

// What prepare() readied for the frame: checked to be this frame at this
// size where the graph prepares anything, and taken (renderer.h).
Renderer::Prepared Renderer::take_prepared(const frame::FrameInputs& inputs, frame::Extent size) {
    if (!prepares()) {
        return Prepared{inputs.index, size, 0u, 0u, false};
    }
    if (!prepared_ || prepared_->index != inputs.index || !(prepared_->size == size)) {
        throw MetalError("Renderer::record: frame " + std::to_string(inputs.index) +
                         " was not prepared at this size (Renderer::prepare)");
    }
    const Prepared taken = *prepared_;
    prepared_.reset();
    return taken;
}

// The frame's constants and its camera framed for `at`, into `ring`'s
// slot; their addresses.
std::pair<MTL::GPUAddress, MTL::GPUAddress> Renderer::write_constants(FrameArray& ring, const FrameSlot& begun,
                                                                      const frame::FrameInputs& inputs,
                                                                      frame::Extent at,
                                                                      std::uint32_t accumulated_frames) {
    const contracts::FrameConstants constants{
        static_cast<float>(inputs.time.count()),
        static_cast<std::uint32_t>(inputs.index),
        at.width,
        at.height,
        accumulated_frames,
        {0u, 0u, 0u},
    };
    const std::span<std::byte> bytes = ring.bytes(begun.slot);
    const MTL::GPUAddress address = ring.address(begun.slot);
    std::memcpy(bytes.data() + constants_offset, &constants, sizeof(constants));
    MTL::GPUAddress camera_address = 0;
    if (inputs.camera) {
        const contracts::CameraData camera = camera::shader_form(*inputs.camera, at);
        std::memcpy(bytes.data() + camera_offset, &camera, sizeof(camera));
        camera_address = address + camera_offset;
    }
    return {address + constants_offset, camera_address};
}

// Every pass's resources at the light size: the constants and camera, the
// scene, the images, and each family's (renderer.h). `previous` whether a
// temporal graph merges last frame.
FrameResources Renderer::frame_resources(const FrameSlot& begun, const frame::FrameInputs& inputs,
                                         MTL::Texture* target, frame::Extent light, const Prepared& prepared,
                                         bool previous) {
    const auto [constants_address, camera_address] =
        write_constants(constants_, begun, inputs, light, prepared.accumulated_frames);
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
        resources.acceleration = acceleration_->resource(begun.slot);
    }
    if (images_) {
        resources.radiance = images_->radiance();
        for (std::uint32_t level = 0; level < passes::bloom_levels; ++level) {
            resources.bloom[level] = images_->bloom(level);
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
    const std::optional<contracts::Camera> previous_camera = previous ? previous_camera_ : std::nullopt;
    if (restir_di_) {
        restir_di_->bind(resources, begun, inputs.index, previous_camera, light);
    }
    if (mnee_) {
        mnee_->bind(resources, begun);
    }
    if (restir_pt_) {
        restir_pt_->bind(resources, begun, inputs.index, previous_camera, light);
    }
    if (features_) {
        features_->bind(resources, begun, inputs.previous, light);
    }
    if (reuses_history_) {
        // Last frame's transforms and glows: this frame's own where the
        // scene is still, else animated by animate() at the previous time.
        resources.last_transforms = last_transforms_ ? last_transforms_->address(begun.slot) : resources.transforms;
        resources.last_glows = last_glows_ ? last_glows_->address(begun.slot) : resources.glows;
    }
    return resources;
}

// Animate: every moving shape placed and every glowing light lit at the
// frame's time, by the core, in the slot's transforms and glows; and, when
// shapes move, the slot's structure rebuilt over them, before any pass
// traces it (renderer.h). With them, last frame's and the features' frame
// before's where the graph keeps them.
void Renderer::animate(MTL4::ComputeCommandEncoder* encoder, const FrameSlot& begun,
                       const frame::FrameInputs& inputs, bool previous) {
    if (!scene_ || !animation::changes(animation_)) {
        return;
    }
    const std::span<contracts::Transform> placed = transforms_->transforms(begun.slot);
    animation::animate(animation_, inputs.time, placed, glows_->glows(begun.slot));
    if (animation::moves(animation_)) {
        acceleration_->update(encoder, begun.slot, placed);
    }
    // Last frame's, at its time; unread at a history's first frame.
    if (last_transforms_ && previous) {
        animation::animate(animation_, previous_time_, last_transforms_->transforms(begun.slot),
                           last_glows_->glows(begun.slot));
    }
    // The features' frame before, at its time; unread without one. Its
    // shapes alone: a glow does not move a surface.
    if (features_ && features_->previous_transforms && inputs.previous) {
        animation::place(animation_, inputs.previous->time, features_->previous_transforms->transforms(begun.slot));
    }
}

// A denoised graph's passes from the denoise pass on (renderer.h): at the
// target's size, with its constants and camera, the radiance image the
// denoise pass writes and the presenting pass reads; the denoise pass's own
// inputs; and the full-size albedo's features dispatch. Precondition: the
// graph denoises.
Renderer::DenoiseResources Renderer::denoise_resources(const FrameResources& resources, const FrameSlot& begun,
                                                       const frame::FrameInputs& inputs, frame::Extent size,
                                                       const Prepared& prepared) {
    DenoiseFamily& d = *denoise_;
    const frame::Extent light = resources.size;
    DenoiseResources out{resources, resources, resources};
    FrameResources& output = out.output;
    const auto [output_constants, output_camera] =
        write_constants(d.output_constants, begun, inputs, size, prepared.accumulated_frames);
    output.constants = output_constants;
    output.camera = output_camera;
    output.size = size;
    output.radiance = images_->radiance();
    output.accumulation = nullptr;

    FrameResources& full_features = out.full_features;
    full_features = output;
    full_features.feature_constants =
        write_feature_constants(d.full_feature_constants, begun.slot, inputs.previous, size);
    full_features.albedo = d.full_features.albedo();
    full_features.normal = d.full_features.normal();
    full_features.depth = d.full_features.depth();
    full_features.motion = d.full_features.motion();

    // Its constants: the view axes, the sizes, whether there is a history.
    const camera::ViewAxes axes = camera::view_axes(*inputs.camera);
    contracts::DenoiseConstants k{};
    k.right = axes.right;
    k.up = axes.up;
    k.backward = axes.backward;
    k.light_width = light.width;
    k.light_height = light.height;
    k.history = prepared.denoise_history ? 1u : 0u;
    k.padded_width = d.images.padded().width;
    k.padded_height = d.images.padded().height;
    k.output_width = size.width;
    k.output_height = size.height;
    k.features = d.features;
    std::memcpy(d.constants.bytes(begun.slot).data(), &k, sizeof(k));

    FrameResources& denoising = out.denoising;
    denoising = output;
    denoising.light_radiance = accumulation_->texture();
    denoising.albedo = resources.albedo;
    denoising.normal = resources.normal;
    denoising.depth = resources.depth;
    denoising.motion = resources.motion;
    denoising.full_albedo = d.full_features.albedo();
    denoising.denoise_constants = d.constants.address(begun.slot);
    denoising.denoise = &d.images;
    return out;
}

// The schedule's passes in order into `encoder`, each after the barriers
// its reads need (renderer.h).
void Renderer::record_steps(MTL4::ComputeCommandEncoder* encoder, const FrameResources& resources,
                            const DenoiseResources& denoised_resources) {
    encoder->setArgumentTable(resources.arguments);
    bool restir_before = false;  // whether a ReSTIR pass came before, in this frame
    bool denoised = false;       // whether the denoise pass came before, in this frame
    for (std::size_t i = 0; i < steps_.size(); ++i) {
        const Step& step = steps_[i];
        if (i == first_cross_frame_) {
            // The frames in flight share the images between passes, the
            // accumulated image and ReSTIR's arrays: none writes them while
            // the frame before still reads them, and none reads the
            // accumulated image before the frame before has written it. One
            // barrier for every such hazard, before the first pass that
            // touches any (GPU.8).
            encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        }
        if (reuses_samples(step.kind) && restir_before) {
            // What the ReSTIR pass before it wrote this frame: the records,
            // path states and reservoirs (GPU.8: a write and its reader).
            barrier_after_dispatch(encoder);
        }
        restir_before = restir_before || reuses_samples(step.kind);
        if (frame::reads_radiance(step.kind)) {
            // The radiance image an earlier pass of this frame wrote.
            barrier_after_dispatch(encoder);
        }
        if (step.kind == frame::PassKind::denoise) {
            // The full-size albedo first, then the pass, both after the
            // barrier above. Only the pass's last dispatch reads the
            // full-size albedo, after barriers of its own (passes/denoise/
            // denoise.h), so its first dispatches overlap this one.
            denoise_->full_features_pass.record(encoder, denoised_resources.full_features);
            std::visit([&](const auto& pass) { pass.record(encoder, denoised_resources.denoising); }, step.pass);
            denoised = true;
            continue;
        }
        const FrameResources& given = denoised ? denoised_resources.output : resources;
        std::visit([&](const auto& pass) { pass.record(encoder, given); }, step.pass);
    }
}

void Renderer::record(const FrameSlot& begun, const frame::FrameInputs& inputs, MTL::Texture* target,
                      frame::Extent size) {
    if (begun.commands == nullptr || begun.slot >= frames_in_flight || target == nullptr || size.width == 0 ||
        size.height == 0) {
        throw MetalError("Renderer::record: no command buffer, no frame slot, no target, or an empty image");
    }
    check_time(inputs.time, inputs.index, "its time");
    const Prepared prepared = take_prepared(inputs, size);
    if (needs_scene_ && !inputs.camera) {
        throw MetalError("Renderer::record: the frame graph reads a scene, and the frame has no camera");
    }
    // The passes before the denoise pass render at the light size, the
    // target's own in a graph without it (renderer.h).
    const frame::Extent light = frame::light_size(schedule_, size);
    // A temporal graph merges last frame when the history holds it and the
    // renderer recorded it (its camera kept): never at a history's first.
    const bool previous = reuses_history_ && prepared.reused_frames > 0 && previous_camera_.has_value();
    const FrameResources resources = frame_resources(begun, inputs, target, light, prepared, previous);

    MTL4::ComputeCommandEncoder* encoder = begun.commands->computeCommandEncoder();
    if (encoder == nullptr) {
        throw MetalError("Renderer::record: the command buffer made no compute encoder");
    }
    animate(encoder, begun, inputs, previous);
    const DenoiseResources denoised = denoise_ ? denoise_resources(resources, begun, inputs, size, prepared)
                                               : DenoiseResources{resources, resources, resources};
    record_steps(encoder, resources, denoised);
    encoder->endEncoding();
    // What the next frame's temporal merge asks of this one.
    previous_camera_ = inputs.camera;
    previous_time_ = inputs.time;
}

std::optional<WindowFrame> render_to_window(Submission& submission, Presenter& presenter, Renderer& renderer,
                                            const frame::FrameInputs& inputs) {
    const auto pool = scoped_pool();
    CA::MetalDrawable* drawable = presenter.acquire();
    if (drawable == nullptr) {
        return std::nullopt;
    }
    // The passes dispatch a thread per pixel of the size asked for, and write
    // the drawable's texture: a drawable of another size would be written
    // out of bounds (I.6).
    const frame::Extent size = presenter.size();
    MTL::Texture* texture = drawable->texture();
    if (texture == nullptr || texture->width() != size.width || texture->height() != size.height) {
        throw MetalError("render_to_window: the drawable is not the " + frame::to_string(size) +
                         " the layer was given");
    }
    renderer.prepare(inputs, size);
    const FrameSlot begun = submission.begin();
    renderer.record(begun, inputs, texture, size);
    submission.present(drawable);
    return WindowFrame{begun.sequence, begun.settled};
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target, Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
    // What recording gets from Metal autoreleased (the frame's encoder) is
    // released at the end of the frame, not whenever the caller's pool, if it
    // has one, drains (P.8).
    const auto pool = scoped_pool();
    renderer.prepare(inputs, target.size());
    const FrameSlot begun = submission.begin();
    renderer.record(begun, inputs, target.texture(), target.size());
    submission.commit();
    return begun.sequence;
}

}  // namespace serenity::metal
