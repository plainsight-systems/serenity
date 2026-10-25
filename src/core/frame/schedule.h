#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/frame/extent.h"
#include "core/frame/frame_inputs.h"
#include "core/passes/denoise.h"
#include "core/passes/restir_pt.h"
#include "core/passes/tone_map.h"
#include "core/sampling/jitter.h"

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
// A pass that computes light (path, restir_pt_shade: writes_radiance())
// writes radiance; the presenting pass, tone_map, reads it and writes the
// target; the test pattern, a diagnostic in display values already, writes
// the target alone. So a frame graph that computes light ends in the
// presenting pass, which gives its light exposure, glare and a film-like
// roll-off (core/passes/tone_map.h). The backend records the barrier
// between a pass that writes an image and a later one that reads it
// (metal/frame/renderer.h).
//
// ReSTIR PT (Lin et al. 2022; ReSTIR PT Enhanced, Lin, Kettunen and Wyman
// 2026) is a family of passes: its unified reservoir holds direct and
// indirect light as whole paths (docs/research/2026-10-11-restir-pt-
// layout.md). Its passes (metal/passes/restir_pt/restir_pt.metal.h states
// the algorithm):
//
//   restir_pt_initial      the path tree at each pixel, NEE at every
//                          vertex, resampled to one path; writes a path
//                          reservoir a pixel (contract 15), the pixel's
//                          primary hit (contract 5's G-buffer, which every
//                          shift to the pixel starts from), and the light
//                          the camera sees directly into the frame's light
//                          buffer.
//   restir_pt_temporal     the temporal reuse: last frame's final path at
//                          where the pixel's surface was, shifted here and
//                          merged by Talbot MIS, its confidence capped;
//                          reads last frame's primary hits, final
//                          reservoirs and duplication map; writes a path
//                          reservoir a pixel, and, with no spatial reuse
//                          after it, the frame's final ones and its estimate
//                          into the light buffer.
//   restir_pt_spatial      the paired spatial reuse: shifts each pixel's
//                          path to its paired neighbors, then merges by
//                          pairwise MIS; reads the path reservoirs; adds
//                          its estimate to the light buffer, and, with
//                          temporal reuse, writes the frame's final
//                          reservoirs.
//   restir_pt_shade        the light buffer (with the initial reservoir's
//                          estimate where no reuse ran) folded into the
//                          accumulated image; accumulates, and so writes
//                          radiance, as the path pass does.
//   restir_pt_duplication  the duplication map: for each pixel, how many of
//                          the frame's final reservoirs in its 17 x 17
//                          neighborhood hold copies of its path, which the
//                          next frame's temporal reuse lowers its cap by.
//
// The order a schedule keeps: restir_pt_initial, restir_pt_temporal if
// present, restir_pt_spatial if present, restir_pt_shade,
// restir_pt_duplication if present, each at most once; initial and shade
// both present if any is; the duplication map only with temporal reuse,
// whose cap is all it serves; and its settings exactly with its passes.
//
// The denoiser's features (contract 17, core/contracts/features.h; docs/
// research/2026-10-14-denoiser-features.md) are one pass:
//
//   features  per pixel, at the frame's size, along the camera's ray the
//             light passes trace there: the albedo and normal of the first
//             surface that is not glass or a mirror, the primary surface's
//             depth and its motion since the frame before; writes the four
//             feature images, which only the denoise pass reads.
//
// A pass of its own, not a part of ReSTIR PT's initial pass, which finds the
// primary hit anyway: so a graph that does not ask for features runs the
// kernels it ran, at their cost and to their bits; any light graph, or none,
// can have them; and they render at a size of their own. It traces its own
// camera ray and the delta surfaces behind it, a few rays a pixel, which a
// light graph that asks for them pays (measured in the note). At most once,
// and before the presenting pass, the last. A graph of the features pass
// alone writes no target, the one graph that may: it is how features are
// rendered at another size than the light, as the training data's are, at
// the full resolution beside a half-resolution light graph
// (headless/options.h, --features). A window refuses it, having nothing to
// show (app/main.cpp).
//
// The denoiser (contract 18, core/contracts/denoise.h; docs/research/
// 2026-10-15-denoiser-live.md) is one pass:
//
//   denoise  Milestone 3's trained network: reads the radiance the light
//            pass wrote and the feature images, at the light size, and
//            writes the frame's radiance at the output size, which the
//            presenting pass after it reads. Keeps its own output as the
//            next frame's history.
//
// A graph with it renders at two sizes: every pass before it at the light
// size, the output's halved each way and rounded up (light_size()), the
// denoise pass and the presenting pass at the output's. So the window's
// light is traced at a quarter of its pixels and the network, not the
// window's layer, upscales it. Its rules (invalid()): at most once; after
// the features pass and after ReSTIR PT's shade (its light is ReSTIR PT's
// accumulated image, the frame's mean, traced checkerboarded, below);
// followed directly by depth_of_field, then the presenting pass, the last;
// with restir_pt_spatial, a sigma the traced lattice can pair at (below);
// and [denoise] settings exactly with it. Its output
// feeds the next frame, so a graph with it reuses history (reuses_history),
// started over as ReSTIR's is: at a history's first frame and when the
// size changes.
//
// Depth of field after the network (contract 21, core/contracts/
// depth_of_field.h; logical-overview.md, principle 12) is one pass:
//
//   depth_of_field  the scene's lens applied to the denoise pass's output:
//                   reads the radiance it wrote and the full-size depth,
//                   blurs each pixel by its circle of confusion, and writes
//                   the radiance the presenting pass shows, in place, at
//                   the output size. No settings: the lens is the camera's
//                   (core/passes/depth_of_field.h).
//
// Its rules (invalid()): at most once, directly after the denoise pass,
// which is followed by it, and directly before the presenting pass. A
// graph with the features pass, the denoised graph and both of the data
// set's graphs among them, traces its light, its features and its
// full-size albedo through a pinhole, the frame's camera with its lens
// taken off (traced_inputs()): its features are rasterized, and a
// rasterizer draws a pinhole's image (docs/research/
// 2026-10-23-apple-silicon-design.md, section 6, change 2), its light the
// same rays (contract 17), as the network is trained on; in the denoised
// graph this pass applies the lens after it. Every other graph traces the
// scene's lens in its rays. So in the demo the lens is drawn once, after
// the network, and the data set's graphs are the network's inputs
// themselves, a pinhole's.
//
// Checkerboard rendering of the light (contract 20, core/contracts/
// checkerboard.h; Milestone 4's Step 4, docs/research/2026-10-17-milestone-
// 4-plan.md) is how the light the network reads is traced, not a pass or
// a setting: in a graph that feeds the network (feeds_network(), below)
// ReSTIR PT's passes trace half the light size's pixels each frame, a
// checkerboard whose fields alternate, and the light ends with the fill,
// which writes the light it hands on whole (to the denoise pass, or in a
// graph without it to the presenting pass and the headless renderer's
// files):
// the traced pixels as they are, each of the others from what the frame
// before traced where its surface was, by its features' motion and depth,
// or from the traced pixels around it (metal/passes/checkerboard_fill/
// checkerboard_fill.h). With restir_pt_spatial, the graph's sigma must
// leave the traced lattice's own at least the least one
// (contracts::min_pairing_sigma times contracts::checkerboard_lattice_step).
// Every other graph traces every pixel.
//
// `last` names the final kind, without being a kind of its own (a second
// name for its value, which no switch needs a case for): schedule.cpp
// asserts its list of every kind against it, so a kind missing from the
// list fails to compile (P.5). A kind is added before `last`, and `last`
// set to it.
enum class PassKind : std::uint8_t {
  test_pattern,  // a diagnostic image, a function of pixel and time; writes the
                 // target
  path,  // path tracing, naive, accumulating while the image holds still; needs
         // a scene; writes radiance
  tone_map,            // radiance exposed, bloomed and rolled off
                       // (core/passes/tone_map.h), then sRGB; writes the target
  restir_pt_initial,   // ReSTIR PT: the path tree, resampled to one path a
                       // pixel; needs a scene
  restir_pt_temporal,  // ReSTIR PT's temporal reuse; needs a scene
  restir_pt_spatial,   // ReSTIR PT's paired spatial reuse; needs a scene
  restir_pt_shade,     // ReSTIR PT's light into the accumulated image;
                       // accumulates; writes radiance
  restir_pt_duplication,  // ReSTIR PT's duplication map, for the next frame's
                          // temporal reuse
  features,  // the denoiser's albedo, normal, depth and motion (contract 17);
             // needs a scene
  denoise,  // the trained network: light and features in, radiance at twice the
            // size out
  depth_of_field,  // the scene's lens, applied after the network from depth
                   // (contract 21)
  last = depth_of_field,
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
bool writes_features(PassKind kind);

// Whether a pass of `kind` is one of ReSTIR PT's (above): it reads or writes
// the path reservoirs or the light buffer, and reads the [restir_pt]
// settings.
bool is_restir_pt(PassKind kind);

struct Schedule {
  std::vector<PassKind> passes;  // run in this order; at least one
  // The tone-map pass's settings: present exactly when the schedule has
  // that pass (graph_file.h, core/passes/tone_map.h).
  std::optional<passes::ToneMap> tone_map;
  // ReSTIR PT's settings: present exactly when the schedule has its
  // passes (graph_file.h, core/contracts/path_reuse_settings.h). None
  // unless given, as every member after it, so a schedule written
  // {passes, tone_map} keeps its meaning.
  std::optional<passes::RestirPt> restir_pt = std::nullopt;
  // The denoiser's settings: present exactly when the schedule has its
  // pass (core/passes/denoise.h).
  std::optional<passes::Denoise> denoise = std::nullopt;
};

// Whether any pass of `schedule` accumulates: whether its frames build on
// the frames before them (frame/history.h).
bool accumulates(const Schedule& schedule);

// Whether `schedule` has restir_pt_temporal or denoise: whether its frames
// build on the reservoirs and primary hits (or the denoiser's output) of
// the frames before them, since its reuse history started
// (frame/history.h, ReuseHistory). Apart from accumulates(): a changing
// scene starts the accumulated image over every frame, and keeps its reuse
// history.
bool reuses_history(const Schedule& schedule);

// Whether `schedule` has the denoise pass, and so renders its light at
// light_size(), traced checkerboarded (above).
bool denoises(const Schedule& schedule);

// The size every pass before the denoise pass renders at, for a frame whose
// output is `output`: the output's, halved each way and rounded up
// (contracts::denoise_scale), in a graph with the denoise pass; the
// output's own in every other graph.
Extent light_size(const Schedule& schedule, Extent output);

// Whether `schedule`'s light is the network's input (logical-overview.md,
// principle 13): it has ReSTIR PT's passes and the features pass, the
// denoised graph live and the data set's input graph
// (graphs/restir_pt_features.toml) offline. Such a graph traces its light
// one way, whichever it is, so the data set's light and features for a
// frame are the denoised graph's for that frame, to the bit
// (tests/gpu/features_test.cpp): checkerboarded and filled (above), through
// a pinhole (traced_inputs()), at the camera's jitter (camera_jitter()).
bool feeds_network(const Schedule& schedule);

// Frame `index`'s camera jitter in `schedule` (core/sampling/jitter.h;
// contracts/frame_constants.h), none where its frames take none: every
// camera ray of a graph that feeds the network passes through the same
// point of its pixel, the frame's offset in a fixed sequence, in place of
// each pixel's own random point, and the network reads the offset. The
// path tracer, the reference, draws its own point in every pixel (the box
// filter it converges to), and the features alone, the data set's
// full-size albedo, as the denoised graph's full-size albedo does: an
// output-resolution image, which production upscalers write unjittered.
std::optional<sampling::CameraJitter> camera_jitter(const Schedule& schedule,
                                                    std::uint64_t index);

// Whether `schedule` applies the camera's lens after the network, by the
// depth_of_field pass, rather than in its rays (above): a denoised graph.
bool lens_after_network(const Schedule& schedule);

// The frame `inputs` as `schedule`'s passes before the depth of field trace
// it (above): in a graph with the features pass, the frame's camera and
// the frame before's with their lens taken off (a radius of 0, a pinhole,
// the rest of each camera as it was); `inputs` itself in every other. Pure
// (F.8): the renderer traces through this and reads the lens for the depth of
// field from `inputs`' own camera (metal/frame/renderer.h).
FrameInputs traced_inputs(const Schedule& schedule, const FrameInputs& inputs);

// Whether `schedule` has a pass that writes the target: every graph but
// the features pass alone (above), which a window refuses.
bool presents(const Schedule& schedule);

// Whether `schedule` has the features pass.
bool writes_features(const Schedule& schedule);

// What makes a schedule one a backend can carry out, decided here, in the
// core, so that every backend refuses the same schedules (principle 10):
//
//   - at least one pass;
//   - at most one pass that accumulates (accumulates()): one accumulated
//     image holds one pass's history, so two such passes, the same kind
//     twice included, would each fold a frame into the other's mean;
//   - one pass writes the target, and it is the last: a frame shows one
//     image, finished by one pass; or, the one exception, the graph is the
//     features pass alone, which shows none (above);
//   - the features pass at most once;
//   - at most one pass writes radiance, every pass that reads it comes after
//     it, and a pass that writes it is followed by one that reads it: light
//     computed and never shown is a mistake, not a frame;
//   - tone-map settings exactly when there is a tone_map pass, with an
//     exposure finite within [-10, 10] stops and a bloom in [0, 1)
//     (core/passes/tone_map.h);
//   - ReSTIR PT's passes in their order, each at most once, with
//     restir_pt_initial and restir_pt_shade both present if any is
//     (above); and its settings exactly with its passes, each setting the
//     passes read in its range and each they do not read its zero
//     (core/contracts/path_reuse_settings.h);
//   - the denoise pass's rules (above), its light ReSTIR PT's at a sigma
//     the traced lattice can pair at, and its settings exactly with it;
//   - the depth of field's rules (above).
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

// ReSTIR PT's settings alone: whether `settings` hold what `passes` read,
// in range, and its zero for what they do not (core/passes/restir_pt.h).
// For the graph reader, to name the line of the setting.
std::optional<std::string> invalid(const passes::RestirPt& settings,
                                   std::span<const PassKind> passes);

// The name each kind is written as in a frame graph file, and back: name()
// is the one mapping, a switch with no default (-Wswitch), and pass_kind()
// reads it back through the list of every kind, which a static_assert holds
// to the enum's length (schedule.cpp), so a kind cannot have a name and be
// missing from the list. A kind no enumerator names is refused by
// std::logic_error here and in every function above, never answered (P.6).
std::string_view name(PassKind kind);
std::optional<PassKind> pass_kind(std::string_view name);

// Every kind, in declaration order: for naming the known ones when a graph
// file names an unknown one.
std::span<const PassKind> all_pass_kinds();

}  // namespace serenity::frame
