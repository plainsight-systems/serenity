#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <vector>

#include "core/frame/frame_inputs.h"
#include "core/frame/history.h"
#include "core/frame/schedule.h"
#include "core/measurement/frame_times.h"
#include "core/output/image_format.h"
#include "core/presentation/offline_renderer.h"
#include "core/scene/scene.h"
#include "headless/options.h"
#include "headless/times.h"

namespace serenity::headless {

// Axis: Presentation (the headless renderer's run).
//
// One run of serenity-headless (main.cpp): what parse(), the graph and the
// plan decided (options.h; core/frame/history.h, plan_headless), and the
// backend's offline renderer (core/presentation/offline_renderer.h), made
// once, for whichever backend the program links; a frame rendered, checked
// and written by its members (F.3), which main() calls in the plan's order.
// It names no backend.
//
// Each frame's time is computed from its index, or frozen by --time, never
// measured; its camera is the scene's; and which samples each frame is, and
// what a converging graph's image holds, is the plan's. So the same command
// writes the same files (principle 1), into a directory that holds nothing
// else (options.h, prepare_output), made once the renderer is, so a graph,
// scene or size the backend refuses makes no directory.
//
// Holds references to main()'s options, schedule and scene, so it is
// neither copied nor moved (C.21). The renderer waits for its frames in
// flight before anything it renders with is released (the interface's
// destructor).
class Run {
 public:
  // Throws what make_offline_renderer() throws, or what prepare_output()
  // does.
  Run(const Options& options, const frame::Schedule& schedule,
      const std::optional<scene::SceneDescription>& loaded,
      const frame::HeadlessPlan& plan, output::ImageSource source);
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;
  Run(Run&&) = delete;
  Run& operator=(Run&&) = delete;
  ~Run() = default;

  // Frame `index`'s samples, each committed, and the last waited for.
  void render(std::uint64_t index);

  // Frame `index`'s samples, each committed, none waited for: a frame
  // --times runs back to back (options.h).
  void commit(std::uint64_t index);

  // Under --times: the instants of every sample settled since the last
  // call added to the run's frame times, and the summary's line written
  // onto `out` each time a second of GPU time has passed (options.h).
  void report_times(std::ostream& out);

  // Refuses frame `index`, rendered, if it left out samples for not being
  // finite, or lost MNEE requests to a full work list.
  void check_finite(std::uint64_t index) const;

  // Frame `index`, rendered, read back and written, with its feature
  // images under --features; each path written onto `out`.
  void write(std::uint64_t index, std::ostream& out);

  // Waits for every frame in flight.
  void finish();

 private:
  std::optional<std::filesystem::path> write_frame(std::uint64_t index);
  std::vector<std::filesystem::path> write_features(std::uint64_t index);
  std::optional<frame::PreviousFrame> previous(std::uint64_t index) const;
  std::uint64_t render_samples(std::uint64_t index);

  const Options& options_;
  const frame::Schedule& schedule_;
  const std::optional<scene::SceneDescription>& loaded_;
  frame::HeadlessPlan plan_;
  output::ImageSource source_;
  std::unique_ptr<presentation::OfflineRenderer> renderer_;
  // The one buffer a written frame is read back into, for its source,
  // allocated once (MEM.9).
  std::vector<std::uint8_t> rgba_;
  std::vector<float> accumulated_;
  // --times: the frames' summary, a second of GPU time a period, and the
  // settled samples taken each frame into one list, made once (MEM.9).
  measurement::FrameTimes times_;
  std::vector<presentation::OfflineRenderer::SettledSample> settled_;
};

}  // namespace serenity::headless
