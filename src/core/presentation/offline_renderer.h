#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "core/contracts/linear_image.h"
#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"

namespace serenity::presentation {

// Axis: Presentation (contract).
//
// The seam between the headless renderer and a backend (file-mapping.md,
// "The headless renderer behind an offline renderer"): what the program
// asks of whichever backend its platform builds, to render a run's frames
// into an image of a fixed size and read them back to write as files. The
// program decides everything else, which frames, which samples, what is
// written and where, refused and printed (headless/run.h); the backend
// renders and reads back. Neither side's types cross it: the frame's
// inputs, a sequence, the core's images and spans of bytes.
//
// One interface of pure virtual functions (I.25, C.121), owned through it
// by the program, so its destructor is public and virtual (C.35, C.127);
// made by make_offline_renderer(), which each backend's library defines and
// the program's executable links one of (C.50: a factory, since which
// backend is the build's, not the caller's), so no file of src/headless/
// names a backend. Not copied or moved through the interface (C.67). One
// thread, the program's.
//
// A frame is rendered, then waited for, then read back: render() commits a
// sample and returns its sequence, wait() blocks until a sequence and every
// one before it has completed, and the reads see every sample committed
// before them. Reading back is the point of this program and is not held
// to a frame's budget (GPU.1); each read fills the caller's memory, made
// once for the run (MEM.9), or, for the feature images, returns them.
//
// Failures throw, each a std::exception with the reason, before anything
// of the frame is committed where the backend can tell (I.10; Metal's
// refusals: core/frame/refusals.h): the destructor waits for every frame
// in flight before anything the GPU may still use is released.
class OfflineRenderer {
 public:
  // A sample's instants on the GPU, as the backend's submission reported
  // them when the sample settled: when its work began and ended, on one
  // timeline, the backend's (host time on Metal, the GPU's own on CUDA), of
  // which only differences mean anything (core/measurement/frame_times.h;
  // TLM.11).
  struct SettledSample {
    std::uint64_t sequence = 0;
    frame::Seconds gpu_start{0.0};
    frame::Seconds gpu_end{0.0};
  };

  // The four feature images (contract 17) as the last frame wrote them.
  struct Features {
    contracts::LinearImage albedo;
    contracts::LinearImage normal;
    contracts::LinearImage depth;
    contracts::LinearImage motion;
  };

  OfflineRenderer(const OfflineRenderer&) = delete;
  OfflineRenderer& operator=(const OfflineRenderer&) = delete;
  OfflineRenderer(OfflineRenderer&&) = delete;
  OfflineRenderer& operator=(OfflineRenderer&&) = delete;
  virtual ~OfflineRenderer() = default;

  // Commits the sample `inputs` is; its sequence, for wait().
  virtual std::uint64_t render(const frame::FrameInputs& inputs) = 0;

  // Blocks until sequence `sequence`, and every one before it, has
  // completed; throws what the GPU's work reported for any of them.
  virtual void wait(std::uint64_t sequence) = 0;

  // Samples left out for not being finite, and MNEE requests lost to a full
  // work list, over every completed sample: each a bug, which the program
  // refuses (headless/run.h); 0 for a graph without the pass that counts.
  virtual std::uint64_t non_finite_samples() const = 0;
  virtual std::uint64_t mnee_lost_requests() const = 0;

  // The displayed image's bytes, R, G, B, A, 8 bits each, rows from the
  // top: displayed_bytes() of them, which read_displayed() fills exactly.
  virtual std::size_t displayed_bytes() const = 0;
  virtual void read_displayed(std::span<std::uint8_t> out) = 0;

  // The accumulated image's floats, RGBA a pixel, rows from the top, at
  // the target's size (contract 6; a denoised graph's radiance image).
  // Throws if the graph has none or `out` is not its size.
  virtual void read_accumulated(std::span<float> out) = 0;

  // The feature images; throws if the graph has no features pass.
  virtual Features read_features() = 0;

  // Blocks until every committed sample has completed.
  virtual void finish() = 0;

  // Appends to `into` the instants of every sample settled since the last
  // call, in the order committed, each once, and forgets them. A caller
  // that takes them each frame keeps the backend's list from growing
  // (MEM.9); one that never asks lets it grow by a few bytes a sample.
  virtual void take_settled(std::vector<SettledSample>& into) = 0;

 protected:
  OfflineRenderer() = default;
};

// The backend the program links, rendering `schedule` over `scene` (null
// when the graph reads none) into an image of `size`. Throws if the backend
// refuses the schedule, the scene or the size (core/frame/refusals.h), or
// cannot make its device.
std::unique_ptr<OfflineRenderer> make_offline_renderer(
    const frame::Schedule& schedule, const scene::SceneDescription* scene,
    frame::Extent size);

}  // namespace serenity::presentation
