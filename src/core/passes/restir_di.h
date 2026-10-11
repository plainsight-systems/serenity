#pragma once

// Axis: Pass (ReSTIR DI's settings: what its passes compute with, for every
// backend).
//
// The numbers ReSTIR DI's five passes (core/frame/schedule.h) are run with,
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
// Each setting, its range, and the value the graphs are written with
// (Bitterli et al. 2020 and NVIDIA's RTXDI use the same, but for the radius,
// RTXDI's 32): named once here, so no number in a pass or a graph's comment
// can disagree with another (ES.45).
//
//   candidates        M, the lights each pixel weighs at its surface
//                     (metal/integrator/restir_di.metal.h, step 2): an
//                     integer in [1, 32], written 8. Its cost per pixel is
//                     M target evaluations and 4 M numbers (a light, a
//                     point on it, and the reservoir's choice); 32 bounds
//                     both. M = 1 is naive next event estimation at the
//                     first surface, with visibility reuse's ray.
//   history           H: the temporal input's count is capped at H x M
//                     (step 4), so a sample chosen long ago is weighed as at
//                     most H frames' candidates: how quickly the lighting
//                     follows a firefly that moves against how much noise
//                     the history removes (logical-overview.md). An integer
//                     in [1, 64], written 20. Read only by a graph with
//                     restir_temporal.
//   neighbors         k, the neighbors merged in (step 5): an integer in
//                     [1, 32], written 5. Each costs one shadow ray and two
//                     target evaluations a pixel; 32 is the width of the
//                     mask of neighbors that passed the similarity test,
//                     which the merge keeps for Z's pass over them
//                     (metal/reuse/spatial.metal.h). Read only by a graph
//                     with restir_spatial.
//   radius            how far a neighbor may be, in pixels of the traced
//                     image: finite, in [1, 1024], written 30 (Bitterli's).
//                     Read only by a graph with restir_spatial.
//   depth_threshold   how far a reused surface's view depth may be from
//                     this one's, relative to this one's: in (0, 1], written
//                     0.1. Read by a graph with restir_temporal or
//                     restir_spatial.
//   normal_threshold  the least cosine between their normals: in [-1, 1],
//                     written 0.5 (-1 accepts any normal). Read as the depth
//                     threshold is.
//
// The thresholds choose which surfaces' reservoirs are reused; they never
// decide correctness. The merges weigh every input by whether its own
// surface could have produced the sample they keep
// (metal/reuse/unbiased.metal.h), so a dissimilar neighbor adds noise, not
// bias (logical-overview.md, "Reservoirs are state keyed on surfaces").
//
// A setting a graph's passes do not read is not written in its graph file
// (graph_file.h: a key nobody reads is an error) and is 0 here, which no
// pass reads. Which settings a schedule's passes read, and their ranges,
// are checked where the graph is read, naming the line
// (core/frame/schedule.h, invalid(); I.5, E.2): a shader trusts them.

#include "core/contracts/shared_layout.h"

namespace serenity {
namespace passes {

SERENITY_CONSTANT uint32_t max_candidates = 32;
SERENITY_CONSTANT uint32_t max_history = 64;
SERENITY_CONSTANT uint32_t max_neighbors = 32;  // a 32-bit mask (above)
SERENITY_CONSTANT float min_radius = 1.0f;
SERENITY_CONSTANT float max_radius = 1024.0f;

// The values the graphs are written with (above).
SERENITY_CONSTANT uint32_t written_candidates = 8;
SERENITY_CONSTANT uint32_t written_history = 20;
SERENITY_CONSTANT uint32_t written_neighbors = 5;
SERENITY_CONSTANT float written_radius = 30.0f;
SERENITY_CONSTANT float written_depth_threshold = 0.1f;
SERENITY_CONSTANT float written_normal_threshold = 0.5f;

// The settings, as the graph gives them and a shader reads them. Plain
// data whose members vary independently (C.2); their ranges are checked
// where they are read (above).
struct RestirDi {
    uint32_t candidates;     // M
    uint32_t history;        // H; 0 when no restir_temporal reads it
    uint32_t neighbors;      // k; 0 when no restir_spatial reads it
    float radius;            // pixels; 0 when no restir_spatial reads it
    float depth_threshold;   // relative; 0 when no merge reads it
    float normal_threshold;  // a cosine; 0 when no merge reads it
    uint32_t padding[2];
};

static_assert(sizeof(RestirDi) == 32, "RestirDi must be the same 32 bytes on the host and in shaders");
#if !defined(__METAL_VERSION__)
static_assert(std::is_trivially_copyable_v<RestirDi>, "RestirDi is written to the GPU as bytes");
#endif

}  // namespace passes
}  // namespace serenity
