#include "metal/frame/renderer.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

#include "core/camera/pinhole.h"
#include "core/contracts/camera.h"
#include "core/contracts/frame_constants.h"
#include "metal/device/error.h"
#include "serenity/metallib/shaders.h"

namespace serenity::metal {

namespace {

// Bytes between the ring's slots, and where in a slot each contract is
// (renderer.h): each aligned for any buffer binding Metal takes, with room
// to grow.
constexpr std::size_t slot_stride = 256;
constexpr std::size_t constants_offset = 0;
constexpr std::size_t camera_offset = 128;
static_assert(constants_offset + sizeof(contracts::FrameConstants) <= camera_offset);
static_assert(camera_offset + sizeof(contracts::CameraData) <= slot_stride);

// Every pass binds at most this many buffers and textures through the table:
// the preview binds the constants, the camera, the structure, the scene's
// thirteen still arrays and the frame's transforms (preview.metal), seventeen
// in all; the path pass a few more for its image's counter.
constexpr NS::UInteger max_buffers = 24;
constexpr NS::UInteger max_textures = 4;

// Shaders read the time as a float (contracts/frame_constants.h): a double
// past float's range would reach them as infinity.
void check_time(const frame::FrameInputs& inputs) {
    const double seconds = inputs.time.count();
    if (!std::isfinite(seconds) || std::abs(seconds) > std::numeric_limits<float>::max()) {
        throw Error("frame " + std::to_string(inputs.index) + ": its time, " + std::to_string(seconds) +
                    " s, is past what the shaders' float holds");
    }
}

std::string describe(const NS::Error* error) {
    if (error == nullptr || error->localizedDescription() == nullptr) {
        return "Metal gave no description";
    }
    return error->localizedDescription()->utf8String();
}

}  // namespace

Renderer::Renderer(const Device& device, Submission& submission, const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : library_(device, serenity::metallib::shaders) {
    // The core decides which schedules can be carried out (principle 10).
    if (const std::optional<std::string> reason = frame::invalid(schedule)) {
        throw Error("Renderer: " + *reason);
    }
    bool accumulates = false;
    bool radiance = false;
    for (frame::PassKind kind : schedule.passes) {
        needs_scene_ = needs_scene_ || frame::needs_scene(kind);
        accumulates = accumulates || frame::accumulates(kind);
        radiance = radiance || frame::writes_radiance(kind);
    }
    if (needs_scene_ && scene == nullptr) {
        throw Error("Renderer: the frame graph's passes read a scene, and none was given (--scene)");
    }
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

    constants_ = NS::TransferPtr(
        device.handle()->newBuffer(slot_stride * frames_in_flight, MTL::ResourceStorageModeShared));
    if (!constants_) {
        throw Error("Renderer: the device made no buffer for the frame constants");
    }
    submission.make_resident(constants_.get());

    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(max_buffers);
    descriptor->setMaxTextureBindCount(max_textures);
    for (auto& table : arguments_) {
        NS::Error* error = nullptr;
        table = NS::TransferPtr(device.handle()->newArgumentTable(descriptor.get(), &error));
        if (!table) {
            throw Error("Renderer: the device made no argument table: " + describe(error));
        }
    }

    kinds_ = schedule.passes;
    passes_.reserve(schedule.passes.size());
    for (frame::PassKind kind : schedule.passes) {
        // No default: a kind this backend does not implement fails the build
        // (core/frame/schedule.h).
        switch (kind) {
        case frame::PassKind::test_pattern:
            passes_.emplace_back(std::in_place_type<TestPatternPass>, device, library_);
            break;
        case frame::PassKind::preview:
            passes_.emplace_back(std::in_place_type<PreviewPass>, device, library_);
            break;
        case frame::PassKind::path:
            passes_.emplace_back(std::in_place_type<PathPass>, device, library_);
            break;
        case frame::PassKind::display:
            passes_.emplace_back(std::in_place_type<DisplayPass>, device, library_);
            break;
        case frame::PassKind::tone_map:
            // The schedule has its settings with its pass (frame::invalid).
            passes_.emplace_back(std::in_place_type<ToneMapPass>, device, library_, submission, *schedule.tone_map);
            break;
        }
    }
    if (radiance) {
        images_ = std::make_unique<FrameImages>(device, submission, true, schedule.tone_map.has_value());
    }
    if (accumulates) {
        accumulation_ = std::make_unique<Accumulation>(device, submission);
        non_finite_ = std::make_unique<NonFinite>(device, submission);
    }

    // Only a graph that reads the scene has it put on the GPU, and only its
    // frames place what moves.
    if (needs_scene_) {
        scene_ = std::make_unique<SceneBuffers>(device, submission, *scene);
        animation_ = scene->animation;
        const bool moves = animation::moves(animation_);
        transforms_ = std::make_unique<ShapeTransforms>(device, submission, scene->shapes.transforms, moves);
        glows_ = std::make_unique<LightGlows>(device, submission, scene->light_counts.spheres,
                                              !animation_.glowers.empty());
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

void Renderer::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
    check_time(inputs);
    if (!accumulation_ && !images_) {
        return;
    }
    prepared_.reset();
    if (images_) {
        images_->prepare(size);
    }
    std::uint32_t held = 0;
    if (accumulation_) {
        held = accumulation_->prepare(inputs, size, animation::changes(animation_));
    }
    prepared_ = Prepared{inputs.index, size, held};
}

std::uint64_t Renderer::non_finite_samples() const {
    return non_finite_ ? non_finite_->count() : 0u;
}

void Renderer::record(const FrameSlot& frame, const frame::FrameInputs& inputs, MTL::Texture* target,
                      frame::Extent size) {
    if (frame.commands == nullptr || target == nullptr || size.width == 0 || size.height == 0) {
        throw Error("Renderer::record: no command buffer, no target, or an empty image");
    }
    check_time(inputs);
    std::uint32_t accumulated_frames = 0;
    if (accumulation_ || images_) {
        if (!prepared_ || prepared_->index != inputs.index || !(prepared_->size == size)) {
            throw Error("Renderer::record: frame " + std::to_string(inputs.index) +
                        " was not prepared at this size (Renderer::prepare)");
        }
        accumulated_frames = prepared_->accumulated_frames;
        prepared_.reset();
    }

    if (needs_scene_ && !inputs.camera) {
        throw Error("Renderer::record: the frame graph reads a scene, and the frame has no camera");
    }

    const contracts::FrameConstants constants{
        static_cast<float>(inputs.time.count()),
        static_cast<std::uint32_t>(inputs.index),
        size.width,
        size.height,
        accumulated_frames,
        {0u, 0u, 0u},
    };
    const std::size_t slot = std::size_t{frame.slot} * slot_stride;
    auto* ring = static_cast<std::byte*>(constants_->contents());
    std::memcpy(ring + slot + constants_offset, &constants, sizeof(constants));

    FrameResources resources;
    resources.arguments = arguments_[frame.slot].get();
    resources.constants = constants_->gpuAddress() + slot + constants_offset;
    resources.target = target;
    resources.size = size;
    if (inputs.camera) {
        const contracts::CameraData camera = camera::shader_form(*inputs.camera, size);
        std::memcpy(ring + slot + camera_offset, &camera, sizeof(camera));
        resources.camera = constants_->gpuAddress() + slot + camera_offset;
    }
    if (scene_) {
        resources.scene = &scene_->addresses();
        resources.transforms = transforms_->address(frame.slot);
        resources.glows = glows_->address(frame.slot);
        resources.acceleration = acceleration_->resource(frame.slot);
    }
    if (images_) {
        resources.radiance = images_->radiance();
        for (std::uint32_t level = 0; level < passes::bloom_levels; ++level) {
            resources.bloom[level] = images_->bloom(level);
        }
    }
    if (accumulation_) {
        resources.accumulation = accumulation_->texture();
        resources.accumulated_frames = accumulated_frames;
        non_finite_->begin_frame(frame.slot, frame.sequence);
        resources.non_finite_counter = non_finite_->address(frame.slot);
    }

    MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
    // Animate: every moving shape placed and every glowing light lit at the
    // frame's time, by the core, in the slot's transforms and glows; and,
    // when shapes move, the slot's structure rebuilt over them, before any
    // pass traces it.
    if (scene_ && animation::changes(animation_)) {
        const std::span<contracts::Transform> placed = transforms_->transforms(frame.slot);
        animation::animate(animation_, inputs.time, placed, glows_->glows(frame.slot));
        if (animation::moves(animation_)) {
            acceleration_->update(encoder, frame.slot, placed);
        }
    }
    encoder->setArgumentTable(resources.arguments);
    for (std::size_t i = 0; i < passes_.size(); ++i) {
        const frame::PassKind kind = kinds_[i];
        if (frame::writes_radiance(kind)) {
            // The frames in flight share the images between passes: none
            // writes them while the frame before still reads them.
            encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        }
        if (frame::reads_radiance(kind)) {
            // The radiance image an earlier pass of this frame wrote.
            encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                               MTL4::VisibilityOptionDevice);
        }
        std::visit([&](const auto& p) { p.record(encoder, resources); }, passes_[i]);
    }
    encoder->endEncoding();
}

std::optional<WindowFrame> render_to_window(Submission& submission, Presenter& presenter, Renderer& renderer,
                                            const frame::FrameInputs& inputs) {
    auto drained = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    CA::MetalDrawable* drawable = presenter.acquire();
    if (drawable == nullptr) {
        return std::nullopt;
    }
    renderer.prepare(inputs, presenter.size());
    const FrameSlot frame = submission.begin();
    renderer.record(frame, inputs, drawable->texture(), presenter.size());
    submission.present(drawable);
    return WindowFrame{frame.sequence, frame.settled};
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target, Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
    renderer.prepare(inputs, target.size());
    const FrameSlot frame = submission.begin();
    renderer.record(frame, inputs, target.texture(), target.size());
    submission.commit();
    return frame.sequence;
}

}  // namespace serenity::metal
