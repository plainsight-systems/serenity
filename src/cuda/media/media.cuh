#pragma once

// Axis: Medium (contract 12's shader side, contracts/medium.h).
//
// The media shapes are filled with, as a shader reads them: the records and
// one array per kind; and the contract's question, transmittance, answered
// by the medium's kind: the one place a shader dispatches on MediumKind. The
// switch has no default, so a kind added to MediumKind and not here
// fails Metal's compile (-Werror), and here, where nvcc does not warn of
// an enumerator a switch leaves out, traps (device/layout.cuh,
// unhandled_kind).
//
// Translated line for line from Metal's metal/media/media.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/medium.h"
#include "core/media/absorbing.h"
#include "core/media/medium.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/media/absorbing.cuh"

namespace serenity {
namespace shaders {

struct Media {
  const serenity::media::MediumRecord* records;
  const serenity::media::AbsorbingData* absorbings;
};

// The share of light, per channel, kept over a stretch `t` long inside
// medium `which`, an index into the records or no_medium (air: 1).
__device__ inline float3 transmittance(Media media, uint which, float t) {
  if (which == serenity::contracts::no_medium) {
    return make_float3(1.0f);
  }
  const serenity::media::MediumRecord record = media.records[which];
  switch (record.kind) {
    case serenity::media::MediumKind::absorbing:
      return absorbing_transmittance(media.absorbings[record.index], t);
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return make_float3(1.0f);
}

}  // namespace shaders
}  // namespace serenity
