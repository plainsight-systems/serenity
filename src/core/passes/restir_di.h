#pragma once

// Axis: Pass (ReSTIR DI's settings: the values its passes run with, as its
// graphs are written).
//
// ReSTIR DI's four passes (core/frame/schedule.h) run with Sample reuse's
// settings (core/contracts/reuse_settings.h, which states each setting, its
// range and what reads it), consumed here as the Pass family's: the settings
// a ReSTIR graph carries are that contract's, so the merges that read them
// (metal/reuse/) depend on the contract and on no Pass. They are decided in
// the core, so every backend reuses the same candidates, history and
// neighbors for the same graph (logical-overview.md, principle 10). They are
// the estimator's values, not its mechanism (change-axes.md: mechanism is
// code, values are data): a graph file gives them (core/frame/graph_file.h),
// every frame the same, and a measurement varies them by writing another
// graph, never by changing code (Per.7). A shader reads them as the contract
// lays them out, inside the reuse constants
// (core/contracts/reuse_constants.h).
//
// What this header owns: the values the graphs are written with, NVIDIA
// RTXDI's shipped defaults (its ReSTIRDI.cpp, read 2026-10-10; the user's
// direction that day: Serenity performs as production software does, and is
// not the paper's reference implementation), each named once here so no
// number in a pass or a graph's comment can disagree with another (ES.45):
//
//   setting              written  RTXDI's name
//   candidates           8        numLocalLightSamples
//   bsdf_candidates      1        numBrdfSamples
//   history              20       maxHistoryLength
//   permutation          on       enablePermutationSampling
//   neighbors            1        numSamples
//   radius               32       samplingRadius
//   depth_threshold      0.1
//   normal_threshold     0.5
//   bias_correction      basic    biasCorrectionMode
//   visibility_reuse     on       reuseFinalVisibility
//   visibility_age       4        finalVisibilityMaxAge
//   visibility_distance  16       finalVisibilityMaxDistance
//
// Every RTXDI default this header does not carry is named, with why, in
// docs/research/2026-10-10-restir-di-layout.md, "Decided": its boiling
// filter (a later, measured addition) among them. Its BSDF candidate,
// first set aside there, is carried since Phase B (2026-10-11): without it
// a light candidate that lands in a near-mirror lobe is the image's rarest,
// largest sample (docs/research/2026-10-11-restir-ris-results.md).

#include "core/contracts/shared_layout.h"
#include "core/contracts/reuse_settings.h"

namespace serenity {
namespace passes {

// ReSTIR DI's settings: Sample reuse's (above), by the name the Frame graph
// and the renderer know them by.
using RestirDi = contracts::ReuseSettings;

// The values the graphs are written with (above).
SERENITY_CONSTANT uint32_t written_candidates = 8u;
SERENITY_CONSTANT uint32_t written_bsdf_candidates = 1u;
SERENITY_CONSTANT uint32_t written_history = 20u;
SERENITY_CONSTANT contracts::Permutation written_permutation = contracts::Permutation::on;
SERENITY_CONSTANT uint32_t written_neighbors = 1u;
SERENITY_CONSTANT float written_radius = 32.0f;
SERENITY_CONSTANT float written_depth_threshold = 0.1f;
SERENITY_CONSTANT float written_normal_threshold = 0.5f;
SERENITY_CONSTANT contracts::BiasCorrection written_bias_correction = contracts::BiasCorrection::basic;
SERENITY_CONSTANT contracts::VisibilityReuse written_visibility_reuse = contracts::VisibilityReuse::on;
SERENITY_CONSTANT uint32_t written_visibility_age = 4u;
SERENITY_CONSTANT float written_visibility_distance = 16.0f;

}  // namespace passes
}  // namespace serenity
