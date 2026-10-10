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

// Shaders read the time as a float (contracts/frame_constants.h): a double
// past float's range would reach them as infinity.
void check_time(const frame::FrameInputs& inputs) {
    const double seconds = inputs.time.count();
    if (!std::isfinite(seconds) || std::abs(seconds) > std::numeric_limits<float>::max()) {
        throw MetalError("frame " + std::to_string(inputs.index) + ": its time, " + std::to_string(seconds) +
                    " s, is past what the shaders' float holds");
    }
}

NS::SharedPtr<MTL4::ArgumentTable> make_argument_table(MTL::Device* device) {
    const auto pool = scoped_pool();  // Metal's error, if any, is autoreleased
    auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
    descriptor->setMaxBufferBindCount(max_buffers);
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
      arguments_(make_argument_table(device.handle())) {
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

    // The first pass that writes the images between passes or accumulates.
    const auto crosses_frames = [](frame::PassKind kind) {
        return frame::writes_radiance(kind) || frame::accumulates(kind);
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
            steps_.push_back(Step{kind, Pass{std::in_place_type<PathPass>, library_}});
            break;
        case frame::PassKind::display:
            steps_.push_back(Step{kind, Pass{std::in_place_type<DisplayPass>, library_}});
            break;
        case frame::PassKind::tone_map:
            // The schedule has its settings with its pass (frame::invalid).
            steps_.push_back(
                Step{kind, Pass{std::in_place_type<ToneMapPass>, device, library_, submission, *schedule.tone_map}});
            break;
        }
    }
    if (radiance) {
        images_ = std::make_unique<FrameImages>(
            device, submission, schedule.tone_map ? FrameImages::Bloom::pyramid : FrameImages::Bloom::none);
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
    if (!accumulation_ && !images_) {
        return;
    }
    prepared_.reset();
    if (images_) {
        images_->prepare(size);
    }
    const std::uint32_t held =
        accumulation_ ? accumulation_->prepare(inputs, size, animation::changes(animation_)) : 0u;
    prepared_ = Prepared{inputs.index, size, held};
}

std::uint64_t Renderer::non_finite_samples() const noexcept {
    return non_finite_ ? non_finite_->count() : 0u;
}

void Renderer::record(const FrameSlot& begun, const frame::FrameInputs& inputs, MTL::Texture* target,
                      frame::Extent size) {
    if (begun.commands == nullptr || begun.slot >= frames_in_flight || target == nullptr || size.width == 0 ||
        size.height == 0) {
        throw MetalError("Renderer::record: no command buffer, no frame slot, no target, or an empty image");
    }
    check_time(inputs);
    std::uint32_t accumulated_frames = 0;
    if (accumulation_ || images_) {
        if (!prepared_ || prepared_->index != inputs.index || !(prepared_->size == size)) {
            throw MetalError("Renderer::record: frame " + std::to_string(inputs.index) +
                        " was not prepared at this size (Renderer::prepare)");
        }
        accumulated_frames = prepared_->accumulated_frames;
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
        resources.accumulation = accumulation_->texture();
        resources.accumulated_frames = accumulated_frames;
        non_finite_->begin_frame(begun.slot, begun.sequence);
        resources.non_finite_counter = non_finite_->address(begun.slot);
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
    }
    encoder->setArgumentTable(resources.arguments);
    for (std::size_t i = 0; i < steps_.size(); ++i) {
        const Step& step = steps_[i];
        if (i == first_cross_frame_) {
            // The frames in flight share the images between passes and the
            // accumulated image: none writes them while the frame before
            // still reads them, and none reads the accumulated image before
            // the frame before has written it. One barrier for both hazards,
            // before the first pass that touches either (GPU.8).
            encoder->barrierAfterQueueStages(MTL::StageDispatch, MTL::StageDispatch, MTL4::VisibilityOptionDevice);
        }
        if (frame::reads_radiance(step.kind)) {
            // The radiance image an earlier pass of this frame wrote.
            encoder->barrierAfterEncoderStages(MTL::StageDispatch, MTL::StageDispatch,
                                               MTL4::VisibilityOptionDevice);
        }
        std::visit([&](const auto& pass) { pass.record(encoder, resources); }, step.pass);
    }
    encoder->endEncoding();
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
