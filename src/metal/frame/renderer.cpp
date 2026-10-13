#include "metal/frame/renderer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "core/camera/thin_lens.h"
#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/mnee_request.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"
#include "core/contracts/reuse_constants.h"
#include "core/contracts/surface_record.h"
#include "core/sampling/pairing.h"
#include "core/scene/packed_range.h"
#include "metal/device/error.h"
#include "metal/device/support.h"
#include "metal/passes/bindings.h"
#include "serenity/metallib/shaders.h"

namespace serenity::metal {

namespace {

// Where in a slot of the ring each contract is (renderer.h): each aligned
// for any buffer binding Metal takes, with room to grow. A slot is one
// FrameArray copy, buffer_alignment bytes.
constexpr std::size_t constants_offset = 0;
constexpr std::size_t camera_offset = 128;
constexpr std::size_t ring_slot_size = buffer_alignment;
static_assert(constants_offset + sizeof(contracts::FrameConstants) <= camera_offset);
static_assert(camera_offset + sizeof(contracts::CameraData) <= ring_slot_size);
static_assert(camera_offset % alignof(contracts::CameraData) == 0);
// Both are written into the ring as their bytes (SL.con.4, COPY.6).
static_assert(std::is_trivially_copyable_v<contracts::FrameConstants>);
static_assert(std::is_trivially_copyable_v<contracts::CameraData>);

// Every pass binds at most this many buffers and textures through the table
// (metal/passes/bindings.h): the path pass binds the most buffers, 0 to 6
// (the constants, the camera, the structure, the scene's block, the frame's
// glows and transforms, and its image's counter), the tone map the most
// textures, 0 to 2. A new kind of material, texture, medium or light adds
// to the block, not here.
constexpr NS::UInteger max_buffers = 8;
constexpr NS::UInteger max_textures = 4;
static_assert(bindings::path::non_finite < max_buffers);
static_assert(bindings::tone_map::target < max_textures);
// A ReSTIR graph's table: its passes bind up to 16 buffers (bindings.h,
// restir_temporal's 0 to 15), so every other graph's table keeps the 8 it
// had (renderer.h).
constexpr NS::UInteger max_restir_buffers = 16;
// A ReSTIR PT graph's: its passes bind up to 26 (bindings.h, restir_pt's 0
// to 25).
constexpr NS::UInteger max_restir_pt_buffers = 28;
// A path graph with mnee: the MNEE pass binds up to 11 (bindings.h, mnee's
// 0 to 10).
constexpr NS::UInteger max_mnee_buffers = 11;
static_assert(bindings::mnee::light < max_mnee_buffers);
static_assert(bindings::restir_pt::mnee_header < max_restir_pt_buffers);
static_assert(bindings::restir_initial::output < max_restir_buffers);
static_assert(bindings::restir_temporal::last_transforms < max_restir_buffers);
static_assert(bindings::restir_spatial::output < max_restir_buffers);
static_assert(bindings::restir_shade::non_finite < max_restir_buffers);


// The reuse textures' seed: fixed, so every run pairs the same pixels
// (core/sampling/pairing.h).
constexpr std::uint64_t pairing_seed = 0x5e2e417f;

// Whether `kind` is a pass of either ReSTIR, whose arrays the frames share.
bool reuses_samples(frame::PassKind kind) {
    return frame::is_restir(kind) || frame::is_restir_pt(kind);
}

// The reuse constants are written into their ring as their bytes.
static_assert(std::is_trivially_copyable_v<contracts::ReuseConstants>);

// What the path state keeps of a medium's index: 16 bits, 0xffff for none
// (core/contracts/surface_record.h). A ReSTIR graph refuses a scene with
// that many media or more (I.5).
constexpr std::size_t max_restir_media = contracts::path_no_medium;

// Shaders read the time as a float (contracts/frame_constants.h): a double
// past float's range would reach them as infinity.
void check_time(const frame::FrameInputs& inputs) {
    const double seconds = inputs.time.count();
    if (!std::isfinite(seconds) || std::abs(seconds) > std::numeric_limits<float>::max()) {
        throw MetalError("frame " + std::to_string(inputs.index) + ": its time, " + std::to_string(seconds) +
                    " s, is past what the shaders' float holds");
    }
}

NS::SharedPtr<MTL4::ArgumentTable> make_argument_table(MTL::Device* device, NS::UInteger buffers) {
    const auto pool = scoped_pool();  // Metal's error, if any, is autoreleased
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(buffers);
    descriptor->setMaxTextureBindCount(max_textures);
    NS::Error* error = nullptr;
    auto table = NS::TransferPtr(device->newArgumentTable(descriptor.get(), &error));
    if (!table) {
        throw MetalError("Renderer: the device made no argument table: " + describe(error));
    }
    return table;
}

}  // namespace

Renderer::Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : submission_(submission),
      library_(device, serenity::metallib::shaders),
      constants_(device, submission, std::vector<std::byte>(ring_slot_size), FrameArray::Copies::per_frame),
      arguments_(make_argument_table(device.handle(), schedule.restir_pt                                       ? max_restir_pt_buffers
                                                      : std::ranges::any_of(schedule.passes, reuses_samples) ? max_restir_buffers
                                                      : schedule.mnee                                        ? max_mnee_buffers
                                                                                                             : max_buffers)) {
    // The core decides which schedules can be carried out (principle 10).
    if (const std::optional<std::string> reason = frame::invalid(schedule)) {
        throw MetalError("Renderer: " + *reason);
    }
    const auto any = [&](bool (*has)(frame::PassKind)) { return std::ranges::any_of(schedule.passes, has); };
    needs_scene_ = any(frame::needs_scene);
    const bool accumulates = any(frame::accumulates);
    const bool radiance = any(frame::writes_radiance);
    if (needs_scene_ && scene == nullptr) {
        throw MetalError("Renderer: the frame graph's passes read a scene, and none was given (--scene)");
    }
    const auto pool = scoped_pool();
    const auto has = [&](frame::PassKind kind) { return std::ranges::find(schedule.passes, kind) != schedule.passes.end(); };
    const bool temporal = has(frame::PassKind::restir_temporal);
    const bool spatial = has(frame::PassKind::restir_spatial);
    const bool path_temporal = has(frame::PassKind::restir_pt_temporal);
    // Raytraced mode's temporal merge traces last frame's surface against
    // last frame's structure, which this backend does not keep yet: refused
    // by name, never traced against this frame's (renderer.h).
    if (temporal && schedule.restir_di &&
        schedule.restir_di->bias_correction == contracts::BiasCorrection::raytraced) {
        throw MetalError("Renderer: the frame graph's restir_temporal pass with bias_correction = \"raytraced\" is "
                         "not implemented by the Metal backend yet: it needs last frame's acceleration structure; "
                         "use \"basic\"");
    }

    // The first pass that writes the images between passes, accumulates,
    // or writes ReSTIR's arrays, which the frame before's shade reads.
    const auto crosses_frames = [](frame::PassKind kind) {
        return frame::writes_radiance(kind) || frame::accumulates(kind) || reuses_samples(kind);
    };
    if (const auto first = std::ranges::find_if(schedule.passes, crosses_frames); first != schedule.passes.end()) {
        first_cross_frame_ = static_cast<std::size_t>(first - schedule.passes.begin());
    }
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
                Step{kind, Pass{std::in_place_type<ToneMapPass>, device, library_, submission, *schedule.tone_map}});
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
        }
    }
    // The pass that accumulates is the one that writes radiance (the core
    // allows one of each, and path is both), and its accumulated image is
    // the frame's radiance (passes/path/path.h): no image of its own then.
    const FrameImages::Radiance own =
        accumulates ? FrameImages::Radiance::accumulated : FrameImages::Radiance::image;
    const FrameImages::Bloom bloom = schedule.tone_map ? FrameImages::Bloom::pyramid : FrameImages::Bloom::none;
    if (radiance && (own == FrameImages::Radiance::image || bloom == FrameImages::Bloom::pyramid)) {
        images_ = std::make_unique<FrameImages>(device, submission, own, bloom);
    }
    if (accumulates) {
        accumulation_ = std::make_unique<Accumulation>(device, submission);
        non_finite_ = std::make_unique<NonFinite>(device, submission);
    }
    // ReSTIR's arrays and constants, only for its graph (renderer.h). Its
    // passes read a scene, which the check above has made sure of.
    if (schedule.restir_di) {
        if (scene->media.size() >= max_restir_media) {
            throw MetalError("Renderer: the scene has " + std::to_string(scene->media.size()) +
                             " media, and ReSTIR DI's path state keeps a medium's index in 16 bits, under " +
                             std::to_string(max_restir_media));
        }
        if (const std::optional<std::string> outside = scene::outside_packed_bsdf(*scene)) {
            throw MetalError("Renderer: the scene's " + *outside);
        }
        restir_ = schedule.restir_di;
        reuses_history_ = temporal;
        reuse_ = std::make_unique<ReuseImages>(device, submission,
                                               temporal ? ReuseImages::Temporal::merged : ReuseImages::Temporal::none,
                                               temporal || spatial ? ReuseImages::Working::needed
                                                                   : ReuseImages::Working::none);
        reuse_constants_ = std::make_unique<FrameArray>(device, submission,
                                                        std::vector<std::byte>(sizeof(contracts::ReuseConstants)),
                                                        FrameArray::Copies::per_frame);
    }

    // ReSTIR PT's arrays, settings and reuse textures, only for its graph.
    if (schedule.restir_pt) {
        // What the G-buffer and the light tiles pack (contracts/
        // primary_hit.h, light_tile.h), refused rather than wrapped (I.5).
        if (scene->media.size() >= contracts::primary_no_medium) {
            throw MetalError("Renderer: the scene has " + std::to_string(scene->media.size()) +
                             " media, and ReSTIR PT's G-buffer keeps a medium's index in 16 bits, under " +
                             std::to_string(contracts::primary_no_medium));
        }
        if (const std::optional<std::string> outside = scene::outside_packed_bsdf(*scene)) {
            throw MetalError("Renderer: the scene's " + *outside);
        }
        if (scene->shapes.records.size() > contracts::tile_largest_id + 1u ||
            scene->light_counts.lights > contracts::tile_largest_id + 1u) {
            throw MetalError("Renderer: the scene has " + std::to_string(scene->shapes.records.size()) +
                             " shapes and " + std::to_string(scene->light_counts.lights) +
                             " lights, and ReSTIR PT's light tiles keep each index in 16 bits, at most " +
                             std::to_string(contracts::tile_largest_id + 1u) + " of either");
        }
        const bool merged = has(frame::PassKind::restir_pt_spatial);
        reuses_history_ = path_temporal;
        path_reuse_ = std::make_unique<PathReuseImages>(
            device, submission, merged ? PathReuseImages::Spatial::merged : PathReuseImages::Spatial::none,
            path_temporal ? PathReuseImages::Temporal::merged : PathReuseImages::Temporal::none);
        if (path_temporal) {
            path_constants_ = std::make_unique<FrameArray>(
                device, submission, std::vector<std::byte>(sizeof(contracts::PathReuseConstants)),
                FrameArray::Copies::per_frame);
        }
        contracts::PathReuseSettings settings = *schedule.restir_pt;
        settings.mnee = schedule.mnee ? 1u : 0u;  // the graph's, for every pass of the family
        const std::vector<std::uint16_t> textures =
            merged ? sampling::reuse_textures(static_cast<double>(settings.sigma), pairing_seed)
                   : std::vector<std::uint16_t>(1, 0u);
        const std::array<std::span<const std::byte>, 2> arrays = {std::as_bytes(std::span(&settings, 1)),
                                                                  std::as_bytes(std::span(textures))};
        path_static_ = std::make_unique<StaticArrays>(device, submission, arrays);
    }

    // MNEE's work list, with mnee; the path pass's own light buffer before
    // its fold (ReSTIR PT's requests add to its light buffer). Its requests
    // keep a shape in 16 bits (contract 16), refused rather than wrapped (I.5).
    if (schedule.mnee) {
        if (scene->shapes.records.size() > contracts::mnee_largest_shape + 1u) {
            throw MetalError("Renderer: the scene has " + std::to_string(scene->shapes.records.size()) +
                             " shapes, and MNEE's requests keep a shape's index in 16 bits, at most " +
                             std::to_string(contracts::mnee_largest_shape + 1u));
        }
        mnee_list_ = std::make_unique<MneeList>(device, submission, schedule.restir_pt ? MneeList::Light::none
                                                                                       : MneeList::Light::own);
    }

    // Only a graph that reads the scene has it put on the GPU, and only its
    // frames place what moves.
    if (needs_scene_) {
        scene_ = std::make_unique<SceneBuffers>(device, submission, *scene);
        animation_ = scene->animation;
        const bool moves = animation::moves(animation_);
        transforms_ = std::make_unique<ShapeTransforms>(
            device, submission, scene->shapes.transforms,
            moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
        // Checked here, not by animate() in a frame, so nothing about the
        // scene can fail once a frame's submission has begun (renderer.h).
        for (const animation::Glower& glower : animation_.glowers) {
            if (glower.target >= scene->light_counts.spheres) {
                throw MetalError("Renderer: a glowing light's target, " + std::to_string(glower.target) +
                            ", is not one of the scene's " + std::to_string(scene->light_counts.spheres) +
                            " sphere lights");
            }
        }
        glows_ = std::make_unique<LightGlows>(device, submission, scene->light_counts.spheres,
                                              !animation_.glowers.empty());
        // Last frame's, for a temporal graph whose scene changes (renderer.h).
        if (reuses_history_ && animation::changes(animation_)) {
            last_transforms_ = std::make_unique<ShapeTransforms>(
                device, submission, scene->shapes.transforms,
                moves ? FrameArray::Copies::per_frame : FrameArray::Copies::one);
            last_glows_ = std::make_unique<LightGlows>(device, submission, scene->light_counts.spheres,
                                                       !animation_.glowers.empty());
        }
        // Where the families meet: the acceleration structure learns which
        // shapes move, by the movers' targets, and nothing else of them.
        std::vector<std::uint32_t> moving;
        moving.reserve(animation_.movers.size());
        for (const animation::Mover& mover : animation_.movers) {
            moving.push_back(mover.target);
        }
        acceleration_ = std::make_unique<SceneAcceleration>(device, submission, scene->shapes, moving);
    }
}

Renderer::~Renderer() {
    // The pipelines and the argument table are not in the residency set, so
    // no Resident waits for them: this does, before any member is released.
    submission_.wait_idle();
}

void Renderer::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
    check_time(inputs);
    if (needs_scene_ && !inputs.camera) {
        throw MetalError("Renderer: frame " + std::to_string(inputs.index) +
                    ": the frame graph reads a scene, and the frame has no camera");
    }
    if (!accumulation_ && !images_ && !reuse_ && !path_reuse_) {
        return;
    }
    prepared_.reset();
    if (images_) {
        images_->prepare(size);
    }
    if (mnee_list_) {
        mnee_list_->prepare(size);
    }
    // 0 without a temporal merge, of either family (never both in a graph).
    const std::uint64_t reused =
        reuse_ ? reuse_->prepare(inputs, size) : path_reuse_ ? path_reuse_->prepare(inputs, size) : 0u;
    const std::uint32_t held =
        accumulation_ ? accumulation_->prepare(inputs, size,
                                       animation::changes(animation_) ? frame::SceneMotion::changing
                                                                      : frame::SceneMotion::still) : 0u;
    prepared_ = Prepared{inputs.index, size, held, reused};
}

std::uint64_t Renderer::non_finite_samples() const noexcept {
    return non_finite_ ? non_finite_->count() : 0u;
}

void Renderer::read_accumulated(std::span<float> out) {
    MTL::Texture* const image = accumulation_ ? accumulation_->texture() : nullptr;
    if (image == nullptr) {
        throw MetalError("read_accumulated: the frame graph accumulates nothing, or no frame has made its image");
    }
    constexpr std::size_t channels = 4;  // RGBA32Float
    const std::size_t row_bytes = std::size_t{image->width()} * channels * sizeof(float);
    const std::size_t floats = std::size_t{image->width()} * image->height() * channels;
    if (out.size() != floats) {
        throw MetalError("read_accumulated: " + std::to_string(out.size()) + " floats given for an image of " +
                         std::to_string(floats));
    }
    const auto pool = scoped_pool();
    auto buffer = NS::TransferPtr(image->device()->newBuffer(row_bytes * image->height(),
                                                             MTL::ResourceStorageModeShared));
    if (!buffer) {
        throw MetalError("read_accumulated: the device made no buffer to read into");
    }
    const Resident resident = submission_.keep_resident(buffer.get());
    const FrameSlot begun = submission_.begin();
    MTL4::ComputeCommandEncoder* const encoder = begun.commands->computeCommandEncoder();
    if (encoder == nullptr) {
        throw MetalError("read_accumulated: the command buffer made no encoder");
    }
    encoder->copyFromTexture(image, 0, 0, MTL::Origin{0, 0, 0}, MTL::Size{image->width(), image->height(), 1},
                             buffer.get(), 0, row_bytes, row_bytes * image->height());
    encoder->endEncoding();
    submission_.commit();
    (void)submission_.wait_until_complete(begun.sequence);
    std::memcpy(out.data(), buffer->contents(), floats * sizeof(float));
}

void Renderer::record(const FrameSlot& begun, const frame::FrameInputs& inputs, MTL::Texture* target,
                      frame::Extent size) {
    if (begun.commands == nullptr || begun.slot >= frames_in_flight || target == nullptr || size.width == 0 ||
        size.height == 0) {
        throw MetalError("Renderer::record: no command buffer, no frame slot, no target, or an empty image");
    }
    check_time(inputs);
    std::uint32_t accumulated_frames = 0;
    std::uint64_t reused_frames = 0;
    if (accumulation_ || images_ || reuse_ || path_reuse_) {
        if (!prepared_ || prepared_->index != inputs.index || !(prepared_->size == size)) {
            throw MetalError("Renderer::record: frame " + std::to_string(inputs.index) +
                        " was not prepared at this size (Renderer::prepare)");
        }
        accumulated_frames = prepared_->accumulated_frames;
        reused_frames = prepared_->reused_frames;
        prepared_.reset();
    }

    if (needs_scene_ && !inputs.camera) {
        throw MetalError("Renderer::record: the frame graph reads a scene, and the frame has no camera");
    }

    const contracts::FrameConstants constants{
        static_cast<float>(inputs.time.count()),
        static_cast<std::uint32_t>(inputs.index),
        size.width,
        size.height,
        accumulated_frames,
        {0u, 0u, 0u},
    };
    const std::span<std::byte> ring = constants_.bytes(begun.slot);
    const MTL::GPUAddress ring_address = constants_.address(begun.slot);
    std::memcpy(ring.data() + constants_offset, &constants, sizeof(constants));

    FrameResources resources{
        .arguments = arguments_.get(),
        .constants = ring_address + constants_offset,
        .target = target,
        .size = size,
    };
    if (inputs.camera) {
        const contracts::CameraData camera = camera::shader_form(*inputs.camera, size);
        std::memcpy(ring.data() + camera_offset, &camera, sizeof(camera));
        resources.camera = ring_address + camera_offset;
    }
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
        resources.accumulated_frames = accumulated_frames;
        non_finite_->begin_frame(begun.slot, begun.sequence);
        resources.non_finite_counter = non_finite_->address(begun.slot);
    }

    // A temporal graph merges last frame when the history holds it and the
    // renderer recorded it (its camera kept): never at a history's first.
    const bool previous = reuses_history_ && reused_frames > 0 && previous_camera_.has_value();
    if (reuse_) {
        // The reuse constants: the settings, and whether there is a frame
        // before to merge, with its camera framed for this frame's size.
        contracts::ReuseConstants reuse{};
        reuse.settings = *restir_;
        reuse.previous = previous ? 1u : 0u;
        if (previous) {
            reuse.previous_camera = camera::shader_form(*previous_camera_, size);
        }
        std::memcpy(reuse_constants_->bytes(begun.slot).data(), &reuse, sizeof(reuse));
        resources.reuse = reuse_constants_->address(begun.slot);
        const ReuseImages::Addresses arrays = reuse_->addresses(inputs.index);
        resources.records = arrays.records;
        resources.path_states = arrays.path_states;
        resources.previous_positions = arrays.previous_positions;
        resources.last_records = arrays.last_records;
        resources.working_reservoirs = arrays.working;
        resources.final_reservoirs = arrays.final_reservoirs;
        resources.last_final_reservoirs = arrays.last_final;
    }
    if (mnee_list_) {
        const MneeList::Addresses list = mnee_list_->addresses();
        resources.mnee_requests = list.requests;
        resources.mnee_header = list.header;
        resources.mnee_grid = list.grid;
        resources.mnee_light = list.light;
    }
    if (path_reuse_) {
        const PathReuseImages::Addresses arrays = path_reuse_->addresses(inputs.index);
        resources.path_settings = path_static_->address(0);
        resources.pairing = path_static_->address(1);
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
        if (path_constants_) {
            // Whether there is a frame before to merge, with its camera
            // framed for this frame's size.
            contracts::PathReuseConstants reuse{};
            reuse.previous = previous ? 1u : 0u;
            if (previous) {
                reuse.previous_camera = camera::shader_form(*previous_camera_, size);
            }
            std::memcpy(path_constants_->bytes(begun.slot).data(), &reuse, sizeof(reuse));
            resources.path_constants = path_constants_->address(begun.slot);
        }
    }
    if (reuses_history_) {
        // Last frame's transforms and glows: this frame's own where the
        // scene is still, else animated below at the previous time.
        resources.last_transforms = last_transforms_ ? last_transforms_->address(begun.slot) : resources.transforms;
        resources.last_glows = last_glows_ ? last_glows_->address(begun.slot) : resources.glows;
    }

    MTL4::ComputeCommandEncoder* encoder = begun.commands->computeCommandEncoder();
    if (encoder == nullptr) {
        throw MetalError("Renderer::record: the command buffer made no compute encoder");
    }
    // Animate: every moving shape placed and every glowing light lit at the
    // frame's time, by the core, in the slot's transforms and glows; and,
    // when shapes move, the slot's structure rebuilt over them, before any
    // pass traces it.
    if (scene_ && animation::changes(animation_)) {
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
    }
    encoder->setArgumentTable(resources.arguments);
    bool restir_before = false;  // whether a ReSTIR pass came before, in this frame
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
            encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                               MTL4::VisibilityOptionDevice);
        }
        restir_before = restir_before || reuses_samples(step.kind);
        if (frame::reads_radiance(step.kind)) {
            // The radiance image an earlier pass of this frame wrote.
            encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                               MTL4::VisibilityOptionDevice);
        }
        std::visit([&](const auto& pass) { pass.record(encoder, resources); }, step.pass);
    }
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
        throw MetalError("render_to_window: the drawable is not the " + std::to_string(size.width) + " x " +
                    std::to_string(size.height) + " the layer was given");
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
