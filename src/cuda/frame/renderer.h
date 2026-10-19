#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/frame/schedule.h"
#include "core/scene/scene.h"
#include "cuda/device/device.h"
#include "cuda/device/offscreen.h"
#include "cuda/device/submission.h"
#include "cuda/passes/test_pattern/test_pattern.h"

namespace serenity::cuda {

// Axis: Frame graph.
//
// Carries out the core's schedule (core/frame/schedule.h) on CUDA: each of
// its passes, in its order, launched on the frame's stream, which orders
// them (no barrier between passes, unlike Metal 4's encoder). It decides
// nothing about what a frame computes; that is the schedule's and the
// core's (logical-overview.md, principle 10): the frame's constants
// (core/frame/frame_constants.h) and its refusals (core/frame/refusals.h)
// are the core's, the same as Metal's.
//
// The backend is built a phase at a time (docs/research/2026-10-19-cuda-
// backend.md), and passes it does not build yet are refused, each by name,
// when the renderer is made, before any frame (the no-facade rule: a graph
// it cannot render never renders a part of itself). Built today: the test
// pattern.
//
// Throws CudaError, with the core's words for its refusals.
class Renderer {
 public:
  // Builds a pass for each entry of `schedule`. `scene` may be null when no
  // pass reads it. Throws CudaError if the schedule is invalid
  // (frame::invalid), a pass reads a scene and there is none, or it has a
  // pass this backend does not build yet.
  Renderer(Submission& submission, const frame::Schedule& schedule,
           const scene::SceneDescription* scene);

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;
  // Waits for the GPU before any pass's module or memory is released
  // (submission.h, Lifetime).
  ~Renderer();

  // Records frame `inputs` on `begun`'s stream, writing `target`. Throws
  // CudaError for a time the shaders' float cannot hold, or a launch that
  // fails.
  void record(const FrameSlot& begun, const frame::FrameInputs& inputs,
              const Offscreen& target);

 private:
  Submission& submission_;
  std::optional<TestPatternPass> test_pattern_;
};

// The first pass of `schedule` this backend does not build yet, refused by
// name: the reason, or none when it builds every one. A pure function of
// the schedule (F.8), so which passes the backend builds is a unit, tested
// as one.
std::optional<std::string> unbuilt_pass(const frame::Schedule& schedule);

// Records frame `inputs` into `target` and commits it, without waiting;
// returns its sequence, for Submission::wait_until_complete().
std::uint64_t render_to_offscreen(Submission& submission, Offscreen& target,
                                  Renderer& renderer,
                                  const frame::FrameInputs& inputs);

}  // namespace serenity::cuda
