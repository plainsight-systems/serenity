#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/passes/tone_map.h"

namespace serenity::frame {

// Axis: Frame graph.
//
// What a frame computes: the passes it runs, in order. Written in a frame
// graph file (core/frame/graph_file.h), carried in the core, and only carried
// out by a backend (logical-overview.md, principle 10), so every backend
// renders the same frame and none can reorder it. The core holds no schedules
// of its own: which passes a frame runs is data (change-axes.md: mechanism is
// code, values are data), kept apart from the scene, so one scene runs under
// any graph.
//
// A pass is named by its kind. Each backend implements each kind once
// (change-axes.md: every shader-side axis exists once per backend), mapping
// the kinds with a switch that has no default, so adding a kind here fails
// every backend's build until that backend implements it or refuses it by
// name. A backend never silently skips a pass.
//
// What each pass reads and writes, of the frame's images (contract 6):
//
//   - the target, what the window shows or the headless renderer writes,
//     in display values: written by the pass that finishes the frame (a
//     presenting pass), and by no other;
//   - the radiance image, the frame's linear radiance at the frame's size,
//     between a pass that computes light and the pass that presents it.
//
// A pass that computes light (preview, path) writes radiance; a presenting
// pass (display, tone_map) reads it and writes the target; the test
// pattern, a diagnostic in display values already, writes the target alone.
// So a frame graph that computes light ends in a presenting pass, which
// decides how its light looks: display shows it as it is, as tests and
// diagnostics want; tone_map gives it exposure, glare and a film-like
// roll-off, as the window and movies want (core/passes/tone_map.h). The backend records
// the barrier between a pass that writes an image and a later one that reads
// it (metal/frame/renderer.h).
enum class PassKind : std::uint8_t {
    test_pattern,  // a diagnostic image, a function of pixel and time; writes the target
    preview,       // deterministic ray tracing, direct light only; needs a scene; writes radiance
    path,          // path tracing, naive, accumulating while the image holds still; needs a scene; writes radiance
    display,       // radiance as it is: the largest channel above 1 scaled to 1, then sRGB; writes the target
    tone_map,      // radiance exposed, bloomed and rolled off (core/passes/tone_map.h), then sRGB; writes the target
};

// Whether a pass of `kind` reads the scene. A frame graph with such a pass
// cannot run without one (metal/frame/renderer.h).
bool needs_scene(PassKind kind);

// Whether a pass of `kind` averages its frames into an accumulated image
// (frame/frame_inputs.h, accumulated_since), so that its frames are a
// function of the frames before them since the image started over.
bool accumulates(PassKind kind);

// Which of the frame's images a pass of `kind` writes and reads; see above.
bool writes_radiance(PassKind kind);
bool reads_radiance(PassKind kind);
bool writes_target(PassKind kind);

struct Schedule {
    std::vector<PassKind> passes;  // run in this order; at least one
    // The tone-map pass's settings: present exactly when the schedule has
    // that pass (graph_file.h, core/passes/tone_map.h).
    std::optional<passes::ToneMap> tone_map;
};

// Whether any pass of `schedule` accumulates: whether its frames build on
// the frames before them (frame/history.h).
bool accumulates(const Schedule& schedule);

// What makes a schedule one a backend can carry out, decided here, in the
// core, so that every backend refuses the same schedules (principle 10):
//
//   - at least one pass;
//   - at most one pass that accumulates (accumulates()): one accumulated
//     image holds one pass's history, so two such passes, the same kind
//     twice included, would each fold a frame into the other's mean;
//   - one pass writes the target, and it is the last: a frame shows one
//     image, finished by one pass;
//   - at most one pass writes radiance, every pass that reads it comes after
//     it, and a pass that writes it is followed by one that reads it: light
//     computed and never shown is a mistake, not a frame;
//   - tone-map settings exactly when there is a tone_map pass, with an
//     exposure finite within [-10, 10] stops and a bloom in [0, 1)
//     (core/passes/tone_map.h).
//
// None if `schedule` is valid; otherwise why not, in words a reader of the
// frame graph file can act on. The graph reader refuses an invalid schedule
// (graph_file.h), and a backend refuses one built otherwise, by this same
// rule (metal/frame/renderer.h).
std::optional<std::string> invalid(const Schedule& schedule);

// The last of those rules alone: whether tone-map `settings` are in range.
// For the graph reader, to name the line of the settings rather than of the
// passes.
std::optional<std::string> invalid(const passes::ToneMap& settings);

// The name each kind is written as in a frame graph file, and back. One table, so
// a kind and its name cannot disagree in two places.
std::string_view name(PassKind kind);
std::optional<PassKind> pass_kind(std::string_view name);

// Every kind, in declaration order: for naming the known ones when a graph
// file names an unknown one.
std::span<const PassKind> all_pass_kinds();

}  // namespace serenity::frame
