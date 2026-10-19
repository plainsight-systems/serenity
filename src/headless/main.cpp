// serenity-headless: renders a frame graph's frames, over a scene if one is
// given, to files of one of Output's kinds, each read back from the source
// that kind is written from (headless/options.h, --format). Each frame's
// time is computed from its index, or frozen by --time, never measured; its
// camera is the scene's; and which samples each frame is, and what a
// converging graph's image holds, is the core's plan (core/frame/history.h,
// plan_headless). So the same command writes the same files (principle 1),
// into a directory that holds nothing else (headless/options.h,
// prepare_output).
//
// Frames are rendered one at a time: each frame's samples are committed, the
// last waited for, and those --write selects are read back and written,
// before the next frame begins. With --features, a written frame's feature
// images are read back and written beside it (headless/options.h).
// Readback is the point of this program, and it is not held to the frame
// budget (GPU.1). A frame that left out samples for not being finite, or
// lost MNEE requests to a full work list (metal/film/failure_count.h), fails
// the run once it has completed, naming the count. Any failure ends the run
// with its message and a non-zero status; the renderer and its target wait
// for the frames in flight before they are released (metal/device/
// submission.h, Lifetime).

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/contracts/linear_image.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/graph_file.h"
#include "core/frame/history.h"
#include "core/frame/schedule.h"
#include "core/output/image_format.h"
#include "core/output/pfm.h"
#include "core/scene/scene.h"
#include "headless/options.h"
#include "metal/device/device.h"
#include "metal/device/offscreen.h"
#include "metal/device/submission.h"
#include "metal/frame/renderer.h"

namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
namespace headless = serenity::headless;
namespace metal = serenity::metal;
namespace output = serenity::output;
namespace scene = serenity::scene;

namespace {

// The arguments after the program's name; none if the system gave not even
// that (POSIX allows argc == 0).
std::span<const char* const> arguments(int argc, char** argv) {
  if (argc < 1) {
    return {};
  }
  return std::span<const char* const>(argv, static_cast<std::size_t>(argc))
      .subspan(1);
}

// One run: what parse(), the graph and the plan decided, and the GPU
// objects it renders and reads back with, made once; a frame rendered and
// written by its members (F.3), which main() calls in the plan's order.
// Holds references to main()'s options, schedule and scene, and its device
// objects, so it is neither copied nor moved (C.21).
class Run {
 public:
  Run(const headless::Options& options, const frame::Schedule& schedule,
      const std::optional<scene::SceneDescription>& loaded,
      const frame::HeadlessPlan& plan, output::ImageSource source)
      : options_(options),
        schedule_(schedule),
        loaded_(loaded),
        plan_(plan),
        source_(source),
        submission_(device_),
        target_(device_, submission_, options.size),
        renderer_(device_, submission_, schedule, loaded ? &*loaded : nullptr) {
    headless::prepare_output(options.out);
    // The one buffer a written frame is read back into, for its source,
    // allocated once (MEM.9). No default: a source with no readback
    // fails to compile (ES.79).
    switch (source_) {
      case output::ImageSource::displayed:
        rgba_.resize(target_.rgba_size());
        break;
      case output::ImageSource::accumulated: {
        // RGBA32Float a pixel (metal/frame/renderer.h, read_accumulated).
        accumulated_.resize(std::size_t{options.size.width} *
                            options.size.height *
                            contracts::accumulated_image_channels);
        break;
      }
    }
  }
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;
  Run(Run&&) = delete;
  Run& operator=(Run&&) = delete;
  ~Run() = default;

  // Frame `index`'s samples, each committed, and the last waited for.
  void render(std::uint64_t index) {
    submission_.wait_until_complete(render_samples(index));
  }

  // Refuses frame `index`, rendered, if it left out samples for not being
  // finite, or lost MNEE requests to a full work list
  // (metal/film/failure_count.h).
  void check_finite(std::uint64_t index) const {
    if (const std::uint64_t failed = renderer_.non_finite_samples();
        failed != 0) {
      throw std::runtime_error(std::format(
          "frame {}: {} samples were not finite, and were left out (a bug)",
          index, failed));
    }
    if (const std::uint64_t lost = renderer_.mnee_lost_requests(); lost != 0) {
      throw std::runtime_error(
          std::format("frame {}: {} MNEE requests found the work list full, "
                      "and their light was lost (a bug)",
                      index, lost));
    }
  }

  // Frame `index`, rendered, read back and written, with its feature
  // images under --features; each path written onto `out`.
  void write(std::uint64_t index, std::ostream& out) {
    if (const std::optional<std::filesystem::path> path = write_frame(index)) {
      out << path->string() << '\n';
    }
    if (options_.features) {
      for (const std::filesystem::path& path : write_features(index)) {
        out << path.string() << '\n';
      }
    }
  }

  // Waits for every frame in flight.
  void finish() { submission_.finish(); }

 private:
  // Frame `index`, read back by its source and written by Output; its
  // path. An accumulated frame is copied three times on its way: GPU
  // to a shared buffer and that into `accumulated_`, 16 bytes a pixel
  // each (read_accumulated), then its radiance into a new image, 12
  // (from_rgba): 33, 33 and 25 MB at 1920 x 1080, once a written frame
  // (GPU.1; docs/research/2026-10-10-reference.md).
  std::optional<std::filesystem::path> write_frame(std::uint64_t index) {
    if (!frame::presents(schedule_)) {
      return std::nullopt;  // nothing shown to write
    }
    std::filesystem::path path =
        headless::frame_path(options_.out, index, options_.format);
    switch (source_) {
      case output::ImageSource::displayed:
        target_.read_rgba(rgba_);
        output::write_image(options_.format, path,
                            output::DisplayedImage{options_.size, rgba_});
        break;
      case output::ImageSource::accumulated:
        renderer_.read_accumulated(accumulated_);
        output::write_image(options_.format, path,
                            contracts::from_rgba(options_.size, accumulated_));
        break;
    }
    return path;
  }

  // Frame `index`'s feature images, read back and written beside it;
  // their paths. Copied as read_features() says (metal/frame/
  // feature_images.h), once a written frame (GPU.1).
  std::vector<std::filesystem::path> write_features(std::uint64_t index) {
    const metal::FeatureImages::Read read = renderer_.read_features();
    // In feature_paths()'s order.
    const std::array<const contracts::LinearImage*, 4> images = {
        &read.albedo, &read.normal, &read.depth, &read.motion};
    const std::array<std::filesystem::path, 4> paths =
        headless::feature_paths(options_.out, index);
    for (std::size_t i = 0; i < paths.size(); ++i) {
      output::write_pfm(paths.at(i), *images.at(i));
    }
    return {paths.begin(), paths.end()};
  }

  // The frame before frame `index` on the timeline, which its features'
  // motion points back to (headless::previous_time), seen by the scene's
  // camera; none without a scene.
  std::optional<frame::PreviousFrame> previous(std::uint64_t index) const {
    const std::optional<frame::Seconds> time =
        headless::previous_time(options_, index);
    if (!loaded_ || !time) {
      return std::nullopt;
    }
    return frame::PreviousFrame{.time = *time, .camera = loaded_->camera};
  }

  // Frame `index`'s samples, each committed; the last one's sequence.
  // The plan has at least one sample a frame (core/frame/history.h).
  std::uint64_t render_samples(std::uint64_t index) {
    const frame::Seconds time = headless::frame_time(options_, index);
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
      last =
          metal::render_to_offscreen(submission_, target_, renderer_, inputs);
    }
    if (!last) {
      throw std::logic_error("frame " + std::to_string(index) +
                             ": the plan gave it no sample");
    }
    return *last;
  }

  const headless::Options& options_;
  const frame::Schedule& schedule_;
  const std::optional<scene::SceneDescription>& loaded_;
  frame::HeadlessPlan plan_;
  output::ImageSource source_;
  // The GPU's, in the order they are made, and released in reverse after
  // the frames in flight (metal/device/submission.h, Lifetime).
  metal::Device device_;
  metal::Submission submission_;
  metal::Offscreen target_;
  metal::Renderer renderer_;
  std::vector<std::uint8_t> rgba_;
  std::vector<float> accumulated_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const headless::Options options = headless::parse(arguments(argc, argv));
    const frame::Schedule schedule = frame::load_schedule(options.graph);
    const output::ImageSource source = output::source(options.format);
    headless::check_graph(options, schedule);
    const std::optional<scene::SceneDescription> loaded =
        options.scene.empty() ? std::nullopt
                              : std::optional{scene::load(options.scene)};

    // Which samples each frame is, and what its image holds, is the
    // core's plan (core/frame/history.h); its samples a frame, not
    // --samples, are what the sample bound is checked at, before
    // anything renders or the directory is made (headless/options.h).
    const frame::HeadlessPlan plan = frame::plan_headless({
        .first = options.first,
        .samples = options.samples,
        .accumulates = frame::accumulates(schedule),
        .scene_changes = loaded && scene::changes(*loaded),
        .time_frozen = options.time.has_value(),
        .reuses = frame::reuses_history(schedule),
    });
    headless::check_samples(options, plan.samples);

    Run run(options, schedule, loaded, plan, source);
    // A graph that reuses history first replays the frames before the
    // run's first, writing none (core/frame/history.h, plan.first).
    for (std::uint64_t index = plan.first; index < plan.first_written;
         ++index) {
      run.render(index);
    }
    // Counted, so the last frame there can be is reachable without its
    // successor (headless/options.h).
    for (std::uint64_t n = 0; n < options.frames; ++n) {
      const std::uint64_t index = options.first + n;
      run.render(index);
      run.check_finite(index);
      if (headless::written(options.write,
                            {.after_first = n, .frames = options.frames})) {
        run.write(index, std::cout);
      }
    }
    run.finish();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "serenity-headless: " << error.what() << '\n';
    return 1;
  }
}
