#pragma once

// The reuse settings. Owned by Sample reuse; read by Pass and the Frame
// graph. Contract 14's sibling, as the reuse constants are
// (contracts/reuse_constants.h, which carries them to the shaders).
//
// The values Sample reuse's rule runs with, and their ranges: how many
// candidates and neighbors it weighs, how much history it keeps, which
// surfaces it takes as similar, how a merge normalizes, and when a
// visibility test is reused (change-axes.md, "Mechanism is code; values are
// data": a sample-reuse rule against how many candidates and neighbors it
// uses). Sample reuse owns them because its code is what reads them: the
// merges (metal/reuse/temporal.metal.h, spatial.metal.h, similar.metal.h),
// their normalization's mode (metal/reuse/unbiased.metal.h) and the shade's
// rule for taking a stored test (metal/reuse/visibility.metal.h), so that
// family depends on this contract and on no Pass. ReSTIR DI's passes run
// with them as a graph writes them (core/passes/restir_di.h: the values the
// graphs are written with, RTXDI's defaults); the Frame graph reads and
// checks them (core/frame/graph_file.h, schedule.h). Decided 2026-10-11, by
// the review of f86e7ae: a mode Sample reuse's merges take had lived in the
// Pass family's header, which the merges then included (change-axes.md:
// families meet at contracts).
//
// The settings, each its meaning and range (the ranges named below, once:
// ES.45):
//
//   candidates        M, the lights each pixel weighs at its surface
//                     (metal/passes/restir_di/restir_di.metal.h, step 2): an
//                     integer in [1, 32]. Its cost per pixel is M target
//                     evaluations and 4 M numbers (a light, a point on it,
//                     and the reservoir's choice); 32 bounds both. M = 1 is
//                     naive next event estimation at the first surface, with
//                     visibility reuse's ray.
//   bsdf_candidates   the candidates each pixel draws by sampling its
//                     surface's BSDF beside the M light candidates (RTXDI's
//                     numBrdfSamples): one ray each, a candidate where it
//                     strikes a light, and both kinds weighed by MIS over the
//                     two strategies' densities (metal/integrator/
//                     target.metal.h, candidate_density). An integer in
//                     [0, 32]; 0 is light candidates alone. Each costs a
//                     BSDF sample, a ray to the nearest surface and four
//                     numbers a pixel, and every light candidate one BSDF
//                     density more where it is not 0. The candidates a
//                     reservoir counts are M plus these.
//   history           H: the temporal input's count is capped at H x M
//                     (step 4), so a sample chosen long ago is weighed as at
//                     most H frames' candidates: how quickly the lighting
//                     follows a firefly that moves against how much noise
//                     the history removes (logical-overview.md). An integer
//                     in [1, 64]. Read only by a graph with restir_temporal.
//   permutation       whether the temporal merge looks up last frame's
//                     reservoir at a permutation of the reprojected pixel
//                     within its aligned 4 x 4 block, the same permutation
//                     for every pixel of a frame and another each frame
//                     (RTXDI's permutation sampling; this permutation is
//                     metal/reuse/temporal.metal.h's), which decorrelates
//                     neighbouring pixels' histories at the cost of reusing
//                     from up to 3 pixels away: off or on. Read only by a
//                     graph with restir_temporal.
//   neighbors         k, the neighbors merged in (step 5): an integer in
//                     [1, 32]. Each costs a gathered record and reservoir
//                     and two target evaluations a pixel, and in raytraced
//                     mode a shadow ray; 32 is the width of the mask of
//                     neighbors that passed the similarity tests, which the
//                     merge keeps for its normalization's walk over them
//                     (metal/reuse/spatial.metal.h). Read only by a graph
//                     with restir_spatial.
//   radius            how far a neighbor may be, in pixels of the traced
//                     image: finite, in [1, 1024]. Read only by a graph with
//                     restir_spatial.
//   depth_threshold   how far a reused surface's view depth may be from
//                     this one's, relative to this one's: in (0, 1]. Read by
//                     a graph with restir_temporal or restir_spatial.
//   normal_threshold  the least cosine between their normals: in [-1, 1]
//                     (-1 accepts any normal). Read as the depth threshold
//                     is.
//   bias_correction   how a merge normalizes what it reuses
//                     (metal/reuse/unbiased.metal.h states both exactly):
//                       basic      RTXDI's Basic. No shadow ray in a merge:
//                                  RTXDI's MIS-like weights over the inputs,
//                                  each by its surface's unshadowed target
//                                  at the survivor. Biased dark where an
//                                  input's surface counts a light it could
//                                  not see: at its worst a light just
//                                  revealed to a surface (a firefly leaving
//                                  a marble's shadow), which history still
//                                  counts as its own, shows M / (M + C) of
//                                  its light the first frame, C the capped
//                                  history (1/21 at RTXDI's defaults,
//                                  core/passes/restir_di.h), recovering as
//                                  history follows. A firefly waking is not
//                                  that case: asleep last frame, it gave
//                                  history nothing. RTXDI's Basic behaves
//                                  alike, and production accepts it;
//                                  raytraced removes it. The shade traces
//                                  the survivor's shadow ray, or reuses a
//                                  recent test (below). 1 shadow ray a
//                                  pixel, and the shade's when it does not
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
//                     Read by a graph with restir_temporal or
//                     restir_spatial; a graph with neither merges nothing,
//                     and is unbiased either way. The Metal backend runs
//                     restir_temporal in basic mode, and refuses by name
//                     raytraced mode with it (last frame's structure is not
//                     yet kept); restir_spatial in either mode
//                     (metal/frame/renderer.h).
//   visibility_reuse  whether the shade, in basic mode, takes a recent
//                     visibility test of its survivor for its own instead of
//                     tracing one (RTXDI's reuseFinalVisibility; step 6;
//                     metal/reuse/visibility.metal.h): off or on. Read only
//                     in basic mode.
//   visibility_age    the oldest test the shade takes, in frames: an integer
//                     in [0, 255]. 0 takes only this frame's. Read where
//                     visibility_reuse is.
//   visibility_distance  the farthest test the shade takes, in pixels of
//                     the traced image, the length of the reservoir's tested
//                     offset (contracts/reservoir.h): finite, in [0, 127]; a
//                     test is taken when its offset is shorter. 127 is the
//                     most the offset holds. Read where visibility_reuse is.
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
//
// Layout rules as for every shared contract (contracts/frame_constants.h).

#include "core/contracts/shared_layout.h"
#include "core/contracts/reservoir.h"

namespace serenity {
namespace contracts {

SERENITY_CONSTANT uint32_t max_candidates = 32;
SERENITY_CONSTANT uint32_t max_bsdf_candidates = 32;
SERENITY_CONSTANT uint32_t max_history = 64;
SERENITY_CONSTANT uint32_t max_neighbors = 32;  // a 32-bit mask (above)
SERENITY_CONSTANT float min_radius = 1.0f;
SERENITY_CONSTANT float max_radius = 1024.0f;
// The most a reservoir's test records (contracts/reservoir.h), one
// definition (ES.3): an age past it saturates, an offset past it clamps.
SERENITY_CONSTANT uint32_t max_visibility_age = tested_max_age;
SERENITY_CONSTANT float max_visibility_distance = float(tested_max_offset);

// How a merge normalizes, and the two switches (above). Types, not bools or
// bare words (I.4); the zero of each is what a graph that does not read it
// holds.
enum class BiasCorrection : uint32_t { basic = 0, raytraced = 1 };
enum class Permutation : uint32_t { off = 0, on = 1 };
enum class VisibilityReuse : uint32_t { off = 0, on = 1 };

// The settings, as a graph gives them and a shader reads them. Plain data
// whose members vary independently (C.2); their ranges are checked where
// they are read (above).
struct ReuseSettings {
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
    uint32_t bsdf_candidates;           // drawn by the BSDF beside the M light candidates
};

static_assert(sizeof(ReuseSettings) == 48, "ReuseSettings must be the same 48 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<ReuseSettings>, "ReuseSettings is written to the GPU as bytes");
#endif

}  // namespace contracts
}  // namespace serenity
