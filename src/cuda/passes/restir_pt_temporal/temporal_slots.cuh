#pragma once

// Axis: Pass (ReSTIR PT's temporal reuse: where its two shifts are kept).
//
// The temporal shifts' two slots of a pixel's shifted paths (step 5 of
// passes/restir_pt/restir_pt.cuh): last frame's path here, and this pixel's
// path there, last frame. Metal's, in its temporal .metal (metal/passes/
// restir_pt_temporal/restir_pt_temporal.metal); here in a header, since
// the shifts that write them are programs and the merge that reads them a
// kernel, two modules.

#include "core/contracts/path_reuse_settings.h"
#include "cuda/passes/restir_pt/restir_pt.cuh"

namespace serenity {
namespace shaders {
namespace temporal {

inline constexpr uint slots = serenity::contracts::max_path_neighbors;
inline constexpr uint last_here = 0u;
inline constexpr uint own_there = 1u;
static_assert(own_there < restir_pt::replay_slots,
              "the temporal shifts fit a pixel's replay slots");

}  // namespace temporal
}  // namespace shaders
}  // namespace serenity
