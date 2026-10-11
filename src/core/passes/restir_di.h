#pragma once

// Axis: Pass (ReSTIR DI's settings: what its passes compute with, for every
// backend).
//
// The numbers ReSTIR DI's four passes (core/frame/schedule.h) are run with,
// decided here, in the core, so every backend reuses the same candidates,
// history and neighbors for the same graph (logical-overview.md,
// principle 10). They are the estimator's values, not its mechanism
// (change-axes.md: mechanism is code, values are data): a graph file gives
// them (core/frame/graph_file.h), every frame the same, and a measurement
// varies them by writing another graph, never by changing code (Per.7). A
// shader reads them as this header lays them out, inside the reuse constants
// (core/contracts/reuse_constants.h), the one definition both sides share
// (contracts/frame_constants.h gives the layout rules).
//
// The values the graphs are written with are NVIDIA RTXDI's shipped
// defaults (its ReSTIRDI.cpp, read 2026-10-10; the user's direction that
// day: Serenity performs as production software does, and is not the
// paper's reference implementation). Each setting, its range, and that
// value, named once here so no number in a pass or a graph's comment can
// disagree with another (ES.45):
//
//   candidates        M, the lights each pixel weighs at its surface
//                     (metal/passes/restir_di/restir_di.metal.h, step 2): an
//                     integer in [1, 32], written 8 (RTXDI's
//                     numLocalLightSamples). Its cost per pixel is M target
//                     evaluations and 4 M numbers (a light, a point on it,
//                     and the reservoir's choice); 32 bounds both. M = 1 is
//                     naive next event estimation at the first surface, with
//                     visibility reuse's ray.
//   history           H: the temporal input's count is capped at H x M
//                     (step 4), so a sample chosen long ago is weighed as at
//                     most H frames' candidates: how quickly the lighting
//                     follows a firefly that moves against how much noise
//                     the history removes (logical-overview.md). An integer
//                     in [1, 64], written 20 (maxHistoryLength). Read only by
//                     a graph with restir_temporal.
//   permutation       whether the temporal merge looks up last frame's
//                     reservoir at a permutation of the reprojected pixel
//                     within its aligned 4 x 4 block, the same permutation
//                     for every pixel of a frame and another each frame
//                     (RTXDI's permutation sampling; this permutation is
//                     metal/reuse/temporal.metal.h's), which decorrelates
//                     neighbouring pixels' histories at the cost of reusing
//                     from up to 3 pixels away: off or on, written on
//                     (enablePermutationSampling). Read only by a graph with
//                     restir_temporal.
//   neighbors         k, the neighbors merged in (step 5): an integer in
//                     [1, 32], written 1 (numSamples). Each costs a gathered
//                     record and reservoir and two target evaluations a
//                     pixel, and in raytraced mode a shadow ray; 32 is the
//                     width of the mask of neighbors that passed the
//                     similarity tests, which the merge keeps for its
//                     normalization's walk over them
//                     (metal/reuse/spatial.metal.h). Read only by a graph
//                     with restir_spatial.
//   radius            how far a neighbor may be, in pixels of the traced
//                     image: finite, in [1, 1024], written 32
//                     (samplingRadius). Read only by a graph with
//                     restir_spatial.
//   depth_threshold   how far a reused surface's view depth may be from
//                     this one's, relative to this one's: in (0, 1], written
//                     0.1. Read by a graph with restir_temporal or
//                     restir_spatial.
//   normal_threshold  the least cosine between their normals: in [-1, 1],
//                     written 0.5 (-1 accepts any normal). Read as the depth
//                     threshold is.
//   bias_correction   how a merge normalizes what it reuses
//                     (metal/reuse/unbiased.metal.h states both exactly):
//                       basic      RTXDI's Basic, the default. No shadow ray
//                                  in a merge: RTXDI's MIS-like weights
//                                  over the inputs, each by its surface's
//                                  unshadowed target at the survivor. Biased
//                                  dark where an input's surface counts a
//                                  light it could not see: at its worst a
//                                  light just revealed to a surface (a
//                                  firefly leaving a marble's shadow), which
//                                  history still counts as its own, shows
//                                  M / (M + C) of its light the first frame,
//                                  C the capped history (1/21 at the
//                                  defaults), recovering as history follows.
//                                  A firefly waking is not that case: asleep
//                                  last frame, it gave history nothing.
//                                  RTXDI's Basic behaves alike, and
//                                  production accepts it; raytraced removes
//                                  it. The shade
//                                  traces the survivor's shadow ray, or
//                                  reuses a recent test (below). 1 shadow ray
//                                  a pixel, and the shade's when it does not
//                                  reuse.
//                       raytraced  the same weights with visibility in
//                                  every input's target, each merge
//                                  tracing the rays that test it (k + 2 a
//                                  pixel). For checking ReSTIR against the
//                                  reference, not for shipping: its mean
//                                  over frames is the reference's image to
//                                  within the light point's and the
//                                  record's quantization (contracts/
//                                  light_sample.h, surface_record.h), a
//                                  tolerance the implementation measures.
//                     Written basic (biasCorrectionMode). Read by a graph
//                     with restir_temporal or restir_spatial; a graph with
//                     neither merges nothing, and is unbiased either way.
//                     Until the merges are implemented (the Metal backend
//                     refuses restir_temporal and restir_spatial by name,
//                     metal/frame/renderer.h), it is read and checked here
//                     and changes no image: no graph that runs merges.
//   visibility_reuse  whether the shade, in basic mode, takes a recent
//                     visibility test of its survivor for its own instead of
//                     tracing one (RTXDI's reuseFinalVisibility; step 6):
//                     off or on, written on. Read only in basic mode.
//   visibility_age    the oldest test the shade takes, in frames: an integer
//                     in [0, 255], written 4 (finalVisibilityMaxAge). 0
//                     takes only this frame's. Read where visibility_reuse
//                     is.
//   visibility_distance  the farthest test the shade takes, in pixels of
//                     the traced image, the length of the reservoir's tested
//                     offset (contracts/reservoir.h): finite, in [0, 127],
//                     written 16 (finalVisibilityMaxDistance); a test is
//                     taken when its offset is shorter. 127 is the most the
//                     offset holds. Read where visibility_reuse is.
//
// Every RTXDI default this header does not carry is named, with why, in
// docs/research/2026-10-10-restir-di-layout.md, "Decided": its boiling
// filter (a later, measured addition), its BSDF candidate and its
// normalization's weights among them.
//
// The similarity thresholds, permutation and the visibility reuse choose
// which surfaces' reservoirs and tests are reused; in raytraced mode they
// never decide correctness: the merges weigh every input by whether its own
// surface could have produced the sample they keep (metal/reuse/
// unbiased.metal.h), so a dissimilar neighbor adds noise, not bias
// (logical-overview.md, "Reservoirs are state keyed on surfaces"). In basic
// mode they also bound its bias: a dissimilar surface's shadows differ more
// from this one's.
//
// A setting a graph's passes do not read is not written in its graph file
// (graph_file.h: a key nobody reads is an error) and is 0 here, which no
// pass reads. Which settings a schedule's passes read, and their ranges,
// are checked where the graph is read, naming the line
// (core/frame/schedule.h, invalid(); I.5, E.2): a shader trusts them.

#include "core/contracts/shared_layout.h"
#include "core/contracts/reservoir.h"

namespace serenity {
namespace passes {

SERENITY_CONSTANT uint32_t max_candidates = 32;
SERENITY_CONSTANT uint32_t max_history = 64;
SERENITY_CONSTANT uint32_t max_neighbors = 32;  // a 32-bit mask (above)
SERENITY_CONSTANT float min_radius = 1.0f;
SERENITY_CONSTANT float max_radius = 1024.0f;
// The most a reservoir's test records (contracts/reservoir.h), one
// definition (ES.3): an age past it saturates, an offset past it clamps.
SERENITY_CONSTANT uint32_t max_visibility_age = contracts::tested_max_age;
SERENITY_CONSTANT float max_visibility_distance = float(contracts::tested_max_offset);

// How a merge normalizes (above). Types, not bools or bare words (I.4); the
// zero of each is what a graph that does not read it holds.
enum class BiasCorrection : uint32_t { basic = 0, raytraced = 1 };
enum class Permutation : uint32_t { off = 0, on = 1 };
enum class VisibilityReuse : uint32_t { off = 0, on = 1 };

// The values the graphs are written with (above).
SERENITY_CONSTANT uint32_t written_candidates = 8;
SERENITY_CONSTANT uint32_t written_history = 20;
SERENITY_CONSTANT Permutation written_permutation = Permutation::on;
SERENITY_CONSTANT uint32_t written_neighbors = 1;
SERENITY_CONSTANT float written_radius = 32.0f;
SERENITY_CONSTANT float written_depth_threshold = 0.1f;
SERENITY_CONSTANT float written_normal_threshold = 0.5f;
SERENITY_CONSTANT BiasCorrection written_bias_correction = BiasCorrection::basic;
SERENITY_CONSTANT VisibilityReuse written_visibility_reuse = VisibilityReuse::on;
SERENITY_CONSTANT uint32_t written_visibility_age = 4;
SERENITY_CONSTANT float written_visibility_distance = 16.0f;

// The settings, as the graph gives them and a shader reads them. Plain
// data whose members vary independently (C.2); their ranges are checked
// where they are read (above).
struct RestirDi {
    uint32_t candidates;                // M
    uint32_t history;                   // H; 0 when no restir_temporal reads it
    Permutation permutation;            // off when no restir_temporal reads it
    uint32_t neighbors;                 // k; 0 when no restir_spatial reads it
    float radius;                       // pixels; 0 when no restir_spatial reads it
    float depth_threshold;              // relative; 0 when no merge reads it
    float normal_threshold;             // a cosine; 0 when no merge reads it
    BiasCorrection bias_correction;     // basic when no merge reads it
    VisibilityReuse visibility_reuse;   // off when not read
    uint32_t visibility_age;            // frames; 0 when not read
    float visibility_distance;          // pixels; 0 when not read
    uint32_t padding;
};

static_assert(sizeof(RestirDi) == 48, "RestirDi must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<RestirDi>, "RestirDi is written to the GPU as bytes");
#endif

}  // namespace passes
}  // namespace serenity
