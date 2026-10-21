#pragma once

// Axis: Frame graph (MNEE's walks between rounds: cuda/passes/mnee/mnee.h).
//
// What the MNEE pass carries from one round of its Newton walks to the
// next (docs/research/2026-10-21-tracing-divergence.md, section 4): a
// walk's state (integrator/mnee.cuh, Walk) and the request it walks for,
// whose chain each round rebuilds from the request rather than carry it
// (mnee.cuh, chain_start). The walks still going after a round are
// appended to the other of two lists, the converged ones to a third, so
// every round's warps hold walks that are still going. And the lists'
// header: each list's count, which the round writing it appends to and
// the pass sets back to 0 before the list is written again.
//
// Compiled by both the host compiler and nvcc, so one type: fixed-width
// fields, packed triples, a C array where device code indexes it (the
// shared layouts' rule, core/contracts/shared_layout.h).

#include <cstdint>

#include "core/contracts/float3.h"
#include "core/contracts/mnee_request.h"

namespace serenity::cuda {

struct MneeWalk {
  std::uint32_t request;    // its index in the request list (contract 16)
  std::uint32_t iteration;  // the walk's iterations taken
  float beta;               // the step's size
  std::uint32_t resolve;    // 1: the next iteration computes the derivatives
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays): device code indexes them
  contracts::Float3 p[contracts::mnee_most_crossings];  // the chain's points
  float dx[contracts::mnee_most_crossings][2];  // the step at the last ones
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays)
};

static_assert(
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers): the layout
    sizeof(MneeWalk) == 96,
    "MneeWalk must be the same 96 bytes on the host and in programs");

// The header's words: the two round lists' counts and the converged
// list's, one 32-bit word each.
inline constexpr std::uint32_t mnee_walk_lists = 2u;
inline constexpr std::uint32_t mnee_converged_word = 2u;
inline constexpr std::uint32_t mnee_walk_header_words = 3u;

// How many iterations a round takes, and so how many rounds a walk of
// at most max_iterations (64, integrator/mnee.cuh) needs: 8 of 8. Eight,
// from the measured walks (the design's section 2): rounds of 8 leave 29%
// of the warps' iterations, of 16 39%.
inline constexpr std::uint32_t mnee_round_iterations = 8u;
inline constexpr std::uint32_t mnee_rounds = 8u;

}  // namespace serenity::cuda
