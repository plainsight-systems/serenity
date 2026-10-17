#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/frame/extent.h"
#include "core/passes/denoise.h"
#include "core/passes/restir_di.h"
#include "core/passes/restir_pt.h"
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
// A pass that computes light (preview, path, restir_shade, restir_pt_shade:
// writes_radiance()) writes radiance; a presenting pass (display, tone_map)
// reads it and writes the target; the test pattern, a diagnostic in display
// values already, writes the target alone.
// So a frame graph that computes light ends in a presenting pass, which
// decides how its light looks: display shows it as it is, as tests and
// diagnostics want; tone_map gives it exposure, glare and a film-like
// roll-off, as the window and movies want (core/passes/tone_map.h). The
// backend records the barrier between a pass that writes an image and a
// later one that reads it (metal/frame/renderer.h).
//
// ReSTIR DI (Bitterli et al. 2020) is four passes, one or more steps of its
// algorithm each (metal/passes/restir_di/restir_di.metal.h states it by
// numbered steps): split where a step must see every pixel's result before
// it, the spatial merge reading its neighbors' reservoirs only after every
// pixel has written its own, and where a pass would otherwise hold more
// alive across its ray queries (logical-overview.md, "Little is kept alive
// across a ray query"). What each reads and writes, of the frame's images
// (contract 6; contracts 5 and 14 lay out the records and the reservoirs):
//
//   restir_initial   the camera's path to its first surface that aims at
//                    lights, then M candidates there and visibility reuse
//                    (steps 1 to 3); writes the surface records, the path
//                    states and, in a graph with a temporal merge, the
//                    surfaces' previous positions; and a reservoir a pixel.
//   restir_temporal  the temporal merge (step 4); reads the records and
//                    previous positions, last frame's records and final
//                    reservoirs, and the initial reservoirs; writes a
//                    reservoir a pixel.
//   restir_spatial   the spatial merge (step 5); reads the records and the
//                    reservoirs the pass before it wrote, at k neighbors;
//                    writes a reservoir a pixel.
//   restir_shade     the direct light from the final reservoir, and the
//                    path's continuation (step 6); reads the records, the
//                    path states and the final reservoirs; accumulates, and
//                    so writes radiance, as the path pass does.
//
// The reservoirs the last pass before restir_shade writes are the frame's
// final reservoirs, which the next frame's temporal merge reads as last
// frame's (metal/frame/frame_images.h). The order a schedule keeps, checked
// by invalid() below: restir_initial, restir_temporal if present,
// restir_spatial if present, restir_shade, each at most once;
// restir_initial and restir_shade both present if any of the four is; and
// the shade, which writes radiance, before the presenting pass that reads
// it, as every light pass is.
//
// ReSTIR PT (Lin et al. 2022; ReSTIR PT Enhanced, Lin, Kettunen and Wyman
// 2026) is its own family of passes, never mixed with ReSTIR DI's in one
// graph: its unified reservoir holds the direct light ReSTIR DI would
// (docs/research/2026-10-11-restir-pt-layout.md). Its passes
// (metal/passes/restir_pt/restir_pt.metal.h states the algorithm):
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
// the features pass and after a light pass that accumulates (its light is
// the accumulated image, the frame's mean); followed by the presenting
// pass alone, directly; and [denoise] settings exactly with it. Its output
// feeds the next frame, so a graph with it reuses history (reuses_history),
// started over as ReSTIR's is: at a history's first frame and when the
// size changes.
//
// Checkerboard rendering of the light (contract 20, core/contracts/
// checkerboard.h; Milestone 4's Step 4, docs/research/2026-10-17-milestone-
// 4-plan.md) is a setting of the frame graph, not a pass: ReSTIR PT's
// passes trace half the light size's pixels each frame, a checkerboard
// whose fields alternate, and the frame's light half ends, where the
// hand-off copied the light's radiance to the image half, with the fill,
// which writes the hand-off whole: the traced pixels as they are, each of
// the others from what the frame before traced where its surface was, by
// its features' motion and depth, or from the traced pixels around it
// (metal/passes/checkerboard_fill/checkerboard_fill.h). Its rules
// (invalid()): only in a graph with ReSTIR PT's passes, which it halves,
// and the denoise pass, whose hand-off the fill writes; and with
// restir_pt_spatial, a sigma that leaves the traced lattice's own at least
// the least one (contracts::min_pairing_sigma times
// contracts::checkerboard_lattice_step). Off unless written, so
// every graph renders what it did.
//
// An estimator is a graph (logical-overview.md, principle 4: estimators are
// interchangeable over the same scene):
//
//   ["restir_initial", "restir_shade"]
//       resampled importance sampling alone, with visibility reuse; with
//       candidates = 1, naive next event estimation at the first surface;
//   with restir_temporal, restir_spatial, or both between them, in that order
//       temporal reuse, spatial reuse, or both;
//   ["path"]
//       the reference, unchanged: the naive path tracer
//       (metal/integrator/path.metal.h), to whose image every ReSTIR graph
//       converges (principle 3).
//
// `last` names the final kind, without being a kind of its own (a second
// name for its value, which no switch needs a case for): schedule.cpp
// asserts its list of every kind against it, so a kind missing from the
// list fails to compile (P.5). A kind is added before `last`, and `last`
// set to it.
enum class PassKind {
    test_pattern,  // a diagnostic image, a function of pixel and time; writes the target
    preview,       // deterministic ray tracing, direct light only; needs a scene; writes radiance
    path,          // path tracing, naive, accumulating while the image holds still; needs a scene; writes radiance
    display,       // radiance as it is: the largest channel above 1 scaled to 1, then sRGB; writes the target
    tone_map,      // radiance exposed, bloomed and rolled off (core/passes/tone_map.h), then sRGB; writes the target
    restir_initial,   // ReSTIR DI steps 1 to 3: the first rough surface, M candidates, visibility; needs a scene
    restir_temporal,  // step 4: last frame's reservoir merged in; needs a scene
    restir_spatial,   // step 5: k neighbors' reservoirs merged in; needs a scene
    restir_shade,     // step 6: the survivor's light and the path's continuation; accumulates; writes radiance
    restir_pt_initial,      // ReSTIR PT: the path tree, resampled to one path a pixel; needs a scene
    restir_pt_temporal,     // ReSTIR PT's temporal reuse; needs a scene
    restir_pt_spatial,      // ReSTIR PT's paired spatial reuse; needs a scene
    restir_pt_shade,        // ReSTIR PT's light into the accumulated image; accumulates; writes radiance
    restir_pt_duplication,  // ReSTIR PT's duplication map, for the next frame's temporal reuse
    features,               // the denoiser's albedo, normal, depth and motion (contract 17); needs a scene
    denoise,                // the trained network: light and features in, radiance at twice the size out
    last = denoise,
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

// Whether a pass of `kind` is one of ReSTIR DI's four (above): it reads or
// writes the surface records or the reservoirs, and reads the [restir_di]
// settings.
bool is_restir(PassKind kind);

// Whether a pass of `kind` is one of ReSTIR PT's (above): it reads or writes
// the path reservoirs or the light buffer, and reads the [restir_pt]
// settings.
bool is_restir_pt(PassKind kind);

struct Schedule {
    std::vector<PassKind> passes;  // run in this order; at least one
    // The tone-map pass's settings: present exactly when the schedule has
    // that pass (graph_file.h, core/passes/tone_map.h).
    std::optional<passes::ToneMap> tone_map;
    // ReSTIR DI's settings: present exactly when the schedule has its
    // passes (graph_file.h, core/contracts/reuse_settings.h). None unless
    // given, as every member after it, so a schedule written {passes,
    // tone_map} keeps its meaning.
    std::optional<passes::RestirDi> restir_di = std::nullopt;
    // ReSTIR PT's settings: present exactly when the schedule has its
    // passes (graph_file.h, core/contracts/path_reuse_settings.h).
    std::optional<passes::RestirPt> restir_pt = std::nullopt;
    // Caustics by manifold next event estimation (metal/integrator/
    // mnee.metal.h), read by the path pass and by ReSTIR PT's passes, one of
    // which a graph that sets it must have; off unless written, so the path
    // tracer every estimator is measured against stays the one it was.
    bool mnee = false;
    // The denoiser's settings: present exactly when the schedule has its
    // pass (core/passes/denoise.h).
    std::optional<passes::Denoise> denoise = std::nullopt;
    // Checkerboard rendering of the light (above): off unless written. Last,
    // so the members before keep their places.
    bool checkerboard = false;
};

// Whether any pass of `schedule` accumulates: whether its frames build on
// the frames before them (frame/history.h).
bool accumulates(const Schedule& schedule);

// Whether `schedule` has restir_temporal, restir_pt_temporal or denoise:
// whether its frames build on the reservoirs and surface records (or
// primary hits) of the frames before them, since its reuse history started
// (frame/history.h, ReuseHistory). Apart from accumulates(): a changing
// scene starts the accumulated image over every frame, and keeps its reuse
// history.
bool reuses_history(const Schedule& schedule);

// Whether `schedule` has the denoise pass, and so renders its light at
// light_size().
bool denoises(const Schedule& schedule);

// The size every pass before the denoise pass renders at, for a frame whose
// output is `output`: the output's, halved each way and rounded up
// (contracts::denoise_scale), in a graph with the denoise pass; the
// output's own in every other graph.
Extent light_size(const Schedule& schedule, Extent output);

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
//   - ReSTIR DI's passes in their order, each at most once, with
//     restir_initial and restir_shade both present if any of the four is
//     (above); and its settings exactly when its passes are present, each
//     setting the passes read, as the graph's other settings decide what
//     they read (bias_correction and visibility_reuse), in its range, and
//     each they do not read its zero (core/contracts/reuse_settings.h);
//   - ReSTIR PT's passes likewise (above), never beside ReSTIR DI's, and
//     its settings exactly with its passes, in range
//     (core/contracts/path_reuse_settings.h);
//   - mnee set only beside a pass that reads it: path, or ReSTIR PT's;
//   - the denoise pass's rules (above), and its settings exactly with it;
//   - checkerboard set only with ReSTIR PT's passes and the denoise pass,
//     and a sigma the traced lattice can pair at (above).
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

// The ReSTIR rule's settings alone: whether `settings` hold what `passes`
// read, in range, and its zero for what they do not (core/passes/
// restir_di.h).
// For the graph reader, to name the line of the setting.
std::optional<std::string> invalid(const passes::RestirDi& settings, std::span<const PassKind> passes);

// ReSTIR PT's settings alone, likewise (core/passes/restir_pt.h).
std::optional<std::string> invalid(const passes::RestirPt& settings, std::span<const PassKind> passes);

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
