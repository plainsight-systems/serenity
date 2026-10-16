#pragma once

// Axis: Pass (ReSTIR PT's settings, as the graphs write them).
//
// ReSTIR PT's settings are contract 15's sibling, Sample reuse's
// (core/contracts/path_reuse_settings.h, which states each and its range),
// as ReSTIR DI's are contract 14's (core/passes/restir_di.h). This header
// names them for the Pass and Frame graph, and keeps the values the graphs
// are written with, each named once with its source (ES.45): ReSTIR PT
// Enhanced's defaults (Lin, Kettunen and Wyman 2026, §7 and Table 1's
// configuration; docs/research/2026-10-11-restir-pt-layout.md, "What is
// built"). A graph file names every setting its passes read
// (core/frame/graph_file.h); these are what graphs/restir_pt.toml writes,
// and what a test of the graph's parsing holds it to.

#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/shared_layout.h"

namespace serenity {
namespace passes {

using RestirPt = contracts::PathReuseSettings;

SERENITY_CONSTANT uint32_t written_nee_candidates = 32;  // Enhanced §7: 32 at the primary hit, 32 / B^2 deeper
SERENITY_CONSTANT uint32_t written_max_surfaces = 9;     // the 2022 code's maxSurfaceBounces
SERENITY_CONSTANT float written_footprint = 0.02f;       // Enhanced §4: c, "to work well in diverse scenes"
SERENITY_CONSTANT float written_roughness = 0.2f;        // the 2022 code's specularRoughnessThreshold
SERENITY_CONSTANT uint32_t written_path_neighbors = 3;   // Enhanced §7: ReSTIR PT's 3 spatial neighbors
SERENITY_CONSTANT float written_sigma = 16.0f;           // Enhanced §7: sigma = sqrt(8 / (9 pi)) R, R = 30
SERENITY_CONSTANT float written_confidence_cap = 20.0f;  // Lin et al. 2022's history; Enhanced §5's c_Cap^Default
SERENITY_CONSTANT float written_cap_floor = 1.0f;        // Enhanced §5: c_Cap^min = 1
SERENITY_CONSTANT float written_cap_exponent = 0.1f;     // Enhanced §5: alpha = 0.1

}  // namespace passes
}  // namespace serenity
