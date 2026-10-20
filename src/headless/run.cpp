#include "headless/run.h"

#include <array>
#include <cstddef>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/contracts/linear_image.h"
#include "core/output/pfm.h"

namespace serenity::headless {

Run::Run(const Options& options, const frame::Schedule& schedule,
         const std::optional<scene::SceneDescription>& loaded,
         const frame::HeadlessPlan& plan, output::ImageSource source)
    : options_(options),
      schedule_(schedule),
      loaded_(loaded),
      plan_(plan),
      source_(source),
      renderer_(presentation::make_offline_renderer(
          schedule, loaded ? &*loaded : nullptr, options.size)),
      times_(times_period) {
  prepare_output(options.out);
  // A frame's settled samples, at most the frames in flight and the one
  // waited for, taken each frame (MEM.9).
  constexpr std::size_t settled_at_once = 4;
  settled_.reserve(settled_at_once);
  // The one buffer a written frame is read back into, for its source. No
  // default: a source with no readback fails to compile (ES.79).
  switch (source_) {
    case output::ImageSource::displayed:
      rgba_.resize(renderer_->displayed_bytes());
      break;
    case output::ImageSource::accumulated: {
      // RGBA32Float a pixel (OfflineRenderer::read_accumulated).
      accumulated_.resize(std::size_t{options.size.width} *
                          options.size.height *
                          contracts::accumulated_image_channels);
      break;
    }
  }
}

void Run::render(std::uint64_t index) {
  renderer_->wait(render_samples(index));
}

void Run::commit(std::uint64_t index) { (void)render_samples(index); }

void Run::report_times(std::ostream& out) {
  renderer_->take_settled(settled_);
  for (const presentation::OfflineRenderer::SettledSample& sample : settled_) {
    times_.add(sample.gpu_start, sample.gpu_end);
    // The period is GPU time: each sample's end is the summary's now.
    if (const std::optional<measurement::Summary> summary =
            times_.take(sample.gpu_end)) {
      out << times_line(options_.size, *summary) << '\n';
    }
  }
  settled_.clear();
}

void Run::check_finite(std::uint64_t index) const {
  if (const std::uint64_t failed = renderer_->non_finite_samples();
      failed != 0) {
    throw std::runtime_error(std::format(
        "frame {}: {} samples were not finite, and were left out (a bug)",
        index, failed));
  }
  if (const std::uint64_t lost = renderer_->mnee_lost_requests(); lost != 0) {
    throw std::runtime_error(
        std::format("frame {}: {} MNEE requests found the work list full, "
                    "and their light was lost (a bug)",
                    index, lost));
  }
}

void Run::write(std::uint64_t index, std::ostream& out) {
  if (const std::optional<std::filesystem::path> path = write_frame(index)) {
    out << path->string() << '\n';
  }
  if (options_.features) {
    for (const std::filesystem::path& path : write_features(index)) {
      out << path.string() << '\n';
    }
  }
}

void Run::finish() { renderer_->finish(); }

// Frame `index`, read back by its source and written by Output; its path.
// An accumulated frame is copied three times on its way: GPU to host, that
// into `accumulated_`, 16 bytes a pixel each (the backend's
// read_accumulated), then its radiance into a new image, 12 (from_rgba):
// 33, 33 and 25 MB at 1920 x 1080, once a written frame (GPU.1;
// docs/research/2026-10-10-reference.md).
std::optional<std::filesystem::path> Run::write_frame(std::uint64_t index) {
  if (!frame::presents(schedule_)) {
    return std::nullopt;  // nothing shown to write
  }
  std::filesystem::path path = frame_path(options_.out, index, options_.format);
  switch (source_) {
    case output::ImageSource::displayed:
      renderer_->read_displayed(rgba_);
      output::write_image(options_.format, path,
                          output::DisplayedImage{options_.size, rgba_});
      break;
    case output::ImageSource::accumulated:
      renderer_->read_accumulated(accumulated_);
      output::write_image(options_.format, path,
                          contracts::from_rgba(options_.size, accumulated_));
      break;
  }
  return path;
}

// Frame `index`'s feature images, read back and written beside it; their
// paths. Once a written frame (GPU.1).
std::vector<std::filesystem::path> Run::write_features(std::uint64_t index) {
  const presentation::OfflineRenderer::Features read =
      renderer_->read_features();
  // In feature_paths()'s order.
  const std::array<const contracts::LinearImage*, 4> images = {
      &read.albedo, &read.normal, &read.depth, &read.motion};
  const std::array<std::filesystem::path, 4> paths =
      feature_paths(options_.out, index);
  for (std::size_t i = 0; i < paths.size(); ++i) {
    output::write_pfm(paths.at(i), *images.at(i));
  }
  return {paths.begin(), paths.end()};
}

// The frame before frame `index` on the timeline, which its features' motion
// points back to (previous_time), seen by the scene's camera; none without a
// scene.
std::optional<frame::PreviousFrame> Run::previous(std::uint64_t index) const {
  const std::optional<frame::Seconds> time = previous_time(options_, index);
  if (!loaded_ || !time) {
    return std::nullopt;
  }
  return frame::PreviousFrame{.time = *time, .camera = loaded_->camera};
}

// Frame `index`'s samples, each committed; the last one's sequence. The plan
// has at least one sample a frame (core/frame/history.h).
std::uint64_t Run::render_samples(std::uint64_t index) {
  const frame::Seconds time = frame_time(options_, index);
  std::optional<std::uint64_t> last;
  for (std::uint64_t s = 0; s < plan_.samples; ++s) {
    const frame::Sample sample = plan_.sample({.frame = index, .sample = s});
    const frame::FrameInputs inputs{
        .time = time,
        .index = sample.index,
        .accumulated_since = sample.accumulated_since,
        .camera = loaded_ ? std::optional{loaded_->camera} : std::nullopt,
        .reused_since = sample.reused_since,
        .previous = previous(index),
    };
    last = renderer_->render(inputs);
  }
  if (!last) {
    throw std::logic_error("frame " + std::to_string(index) +
                           ": the plan gave it no sample");
  }
  return *last;
}

}  // namespace serenity::headless
