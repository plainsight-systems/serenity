#pragma once

// Axis: GPU backend (device code).
//
// Reading the core's shared layouts in device code: a contracts::Float3 is
// the core's twelve packed bytes, CUDA's float3 the type the arithmetic of
// math/vector.cuh is over (twelve bytes too, but not the same type), so
// device code makes one from the other where it reads or writes it
// (contracts/float3.h), as Metal's shaders do (metal/device/
// layout.metal.h); and a signed byte packed into a word.
//
// Translated line for line from Metal's metal/device/layout.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/float3.h"
#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

__device__ inline float3 to_float3(serenity::contracts::Float3 a) {
  return make_float3(a.x, a.y, a.z);
}

__device__ inline serenity::contracts::Float3 to_packed(float3 a) {
  return serenity::contracts::Float3{a.x, a.y, a.z};
}

// The signed 8-bit value at bit `shift` of `word`, as layouts pack one: its
// bits taken apart as unsigned (ES.101) and read back as two's complement.
__device__ inline int signed_byte(uint word, uint shift) {
  return static_cast<int>(static_cast<signed char>(
      static_cast<unsigned char>((word >> shift) & 0xffu)));
}

// After a switch over a kind with no default (the shapes', materials',
// textures', media's, lights' and BSDFs' dispatches): a kind no case
// handled. Metal refuses such a switch at compile time (-Werror on its
// -Wswitch); nvcc does not warn of an enumerator a switch leaves out
// (checked 2026-10-19), so device code refuses it when it runs: a trap,
// which ends the launch and which the submission's next wait reports as a
// fault (cuda/device/error.h), rather than an answer for a kind it does not
// know (the no-facade rule). Unreached for every kind the core has.
__device__ inline void unhandled_kind() { __trap(); }

}  // namespace shaders
}  // namespace serenity
