// serenity-headless: renders a frame graph's frames, over a scene if one is
// given, to files of one of Output's kinds, each read back from the source
// that kind is written from (headless/options.h, --format), on whichever
// backend the program is linked with (core/presentation/
// offline_renderer.h): it names none. What a run renders, checks and writes
// is headless/run.h's; this file parses the command line, loads the graph
// and the scene, makes the core's plan, and runs it.
//
// Frames are rendered one at a time: each frame's samples are committed, the
// last waited for, and those --write selects are read back and written,
// before the next frame begins. With --features, a written frame's feature
// images are read back and written beside it (headless/options.h).
// Readback is the point of this program, and it is not held to the frame
// budget (GPU.1). A frame that left out samples for not being finite, or
// lost MNEE requests to a full work list, fails the run once it has
// completed, naming the count. Under --times, a frame not written is
// committed and not waited for, so frames run back to back, and the
// frame-time summary is printed each second of GPU time (headless/
// options.h); the failure counts are checked each frame over the frames
// completed so far, and once more after the last. Any failure ends the run with
// its message and a non-zero status; the renderer waits for the frames in
// flight before what it renders with is released.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <optional>
#include <span>

#include "core/frame/graph_file.h"
#include "core/frame/history.h"
#include "core/frame/schedule.h"
#include "core/output/image_format.h"
#include "core/scene/scene.h"
#include "headless/options.h"
#include "headless/run.h"

namespace frame = serenity::frame;
namespace headless = serenity::headless;
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

    headless::Run run(options, schedule, loaded, plan, source);
    // A graph that reuses history first replays the frames before the
    // run's first, writing none (core/frame/history.h, plan.first).
    // Under --times they are frames of the timeline like the rest,
    // committed back to back and measured.
    for (std::uint64_t index = plan.first; index < plan.first_written;
         ++index) {
      if (options.times) {
        run.commit(index);
        run.report_times(std::cout);
      } else {
        run.render(index);
      }
    }
    // Counted, so the last frame there can be is reachable without its
    // successor (headless/options.h).
    for (std::uint64_t n = 0; n < options.frames; ++n) {
      const std::uint64_t index = options.first + n;
      const bool written = headless::written(
          options.write, {.after_first = n, .frames = options.frames});
      if (options.times && !written) {
        run.commit(index);
      } else {
        run.render(index);
      }
      run.check_finite(index);
      if (written) {
        run.write(index, std::cout);
      }
      if (options.times) {
        run.report_times(std::cout);
      }
    }
    run.finish();
    if (options.times) {
      run.check_finite(options.first + options.frames - 1);
      run.report_times(std::cout);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "serenity-headless: " << error.what() << '\n';
    return 1;
  }
}
