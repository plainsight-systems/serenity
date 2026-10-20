#include "cuda/frame/renderer.h"

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <vector>

#include "core/camera/thin_lens.h"
#include "core/frame/frame_constants.h"
#include "core/frame/refusals.h"
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

// The pass kinds this backend builds (renderer.h).
bool built(frame::PassKind kind) {
  return kind == frame::PassKind::test_pattern ||
         kind == frame::PassKind::path || kind == frame::PassKind::tone_map;
}

}  // namespace

std::optional<std::string> unbuilt_pass(const frame::Schedule& schedule) {
  for (const frame::PassKind kind : schedule.passes) {
    if (!built(kind)) {
      return "the CUDA backend has no " + std::string(name(kind)) +
             " pass yet (docs/research/2026-10-19-cuda-backend.md, its "
             "phases 4 and 5)";
    }
  }
  return std::nullopt;
}

Renderer::Renderer(const Device& device, Submission& submission,
                   const frame::Schedule& schedule,
                   const scene::SceneDescription* scene)
    : submission_(submission),
      schedule_(schedule),
      needs_scene_(std::ranges::any_of(schedule.passes, frame::needs_scene)) {
  // The core decides which schedules can be carried out (principle 10).
  refuse(frame::invalid(schedule));
  refuse(frame::scene_refusal(schedule, scene));
  refuse(unbuilt_pass(schedule));
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
  for (const frame::PassKind kind : schedule.passes) {
    switch (kind) {
      case frame::PassKind::test_pattern:
        test_pattern_.emplace();
        break;
      case frame::PassKind::path: {
        const std::array<Pipeline::HitData, structure_records> records =
            acceleration_->hit_records(scene_->block(), transforms_->address());
        path_.emplace(*optix_, submission_, records);
        break;
      }
      case frame::PassKind::tone_map:
        // frame::invalid refuses a tone map without its settings.
        if (!schedule.tone_map) {
          throw CudaError("Renderer: a tone map with no settings");
        }
        tone_map_.emplace(*schedule.tone_map);
        break;
      default:
        break;  // refused above (unbuilt_pass)
    }
  }
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
  acceleration_ = std::make_unique<SceneAcceleration>(
      *optix_, submission_, scene.shapes,
      SceneAcceleration::Roles{.moving = moving, .lights = lights});
}

Renderer::~Renderer() { submission_.wait_idle(); }

void Renderer::prepare(const frame::FrameInputs& inputs, frame::Extent size) {
  refuse(frame::time_refusal(inputs.time, inputs.index, "its time"), "");
  refuse(frame::camera_refusal(inputs, needs_scene_));
  prepared_.reset();
  if (images_) {
    images_->prepare(size);
  }
  std::uint32_t held = 0;
  if (accumulation_) {
    const frame::SceneMotion motion = animation::changes(animation_)
                                          ? frame::SceneMotion::changing
                                          : frame::SceneMotion::still;
    held = accumulation_->prepare(inputs, size, motion);
  }
  prepared_ = Prepared{inputs.index, size, held};
}

void Renderer::animate(const FrameSlot& begun,
                       const frame::FrameInputs& inputs) {
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
}

PathLaunch Renderer::path_launch(const FrameSlot& begun,
                                 const frame::FrameInputs& inputs,
                                 const Prepared& prepared,
                                 frame::Extent size) const {
  const PitchedImage* image = accumulation_->image();
  if (image == nullptr || !inputs.camera) {
    throw CudaError(
        "Renderer: the path pass has no accumulated image or no camera");
  }
  return PathLaunch{
      .frame =
          frame::frame_constants(inputs, size, prepared.accumulated_frames,
                                 frame::camera_jitter(schedule_, inputs.index)),
      .camera = camera::shader_form(*inputs.camera, size),
      .structures = acceleration_->block(),
      .scene = scene_->block_address(),
      .transforms = transforms_->address(),
      .glows = glows_->address(),
      .non_finite = non_finite_->address(begun.slot),
      .accumulated = image->address(),
      .pitch = image->pitch(),
  };
}

void Renderer::record(const FrameSlot& begun, const frame::FrameInputs& inputs,
                      const Offscreen& target) {
  const frame::Extent size = target.size();
  if (!prepared_ || prepared_->index != inputs.index ||
      !(prepared_->size == size)) {
    throw CudaError("Renderer::record: frame " + std::to_string(inputs.index) +
                    " was not prepared at this size (Renderer::prepare)");
  }
  const Prepared prepared = *prepared_;
  prepared_.reset();

  animate(begun, inputs);
  if (non_finite_) {
    non_finite_->begin_frame(begun);
  }
  for (const frame::PassKind kind : schedule_.passes) {
    switch (kind) {
      case frame::PassKind::test_pattern:
        made(test_pattern_, "test pattern")
            .record(
                begun.stream,
                frame::frame_constants(
                    inputs, size, prepared.accumulated_frames, std::nullopt),
                target.address());
        break;
      case frame::PassKind::path:
        made(path_, "path")
            .record(begun, path_launch(begun, inputs, prepared, size));
        break;
      case frame::PassKind::tone_map:
        made(tone_map_, "tone map")
            .record(begun.stream, *accumulation_->image(), *images_,
                    target.address());
        break;
      default:
        break;  // refused at construction (unbuilt_pass)
    }
  }
  if (non_finite_) {
    non_finite_->end_frame(begun);
  }
}

std::uint64_t Renderer::non_finite_samples() const noexcept {
  return non_finite_ ? non_finite_->count() : 0u;
}

void Renderer::read_accumulated(std::span<float> out) const {
  if (!accumulation_ || accumulation_->image() == nullptr) {
    throw CudaError(
        "Renderer::read_accumulated: the graph keeps no accumulated image, "
        "or none has been rendered");
  }
  accumulation_->image()->read(out);
}

std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs) {
  renderer.prepare(inputs, target.size());
  const FrameSlot begun = submission.begin();
  renderer.record(begun, inputs, target);
  submission.commit();
  return begun.sequence;
}

}  // namespace serenity::cuda
