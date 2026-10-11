#pragma once

// Axis: Sample reuse (the reservoir and its merge rule).
//
// Resampled importance sampling one sample at a time (Talbot et al. 2005;
// Bitterli et al. 2020, Algorithms 2 and 3): a stream of weighted samples
// goes in, one comes out, chosen with probability its weight over the
// stream's total, and a running total is all that is kept besides it. The
// one rule every reuse merges by, candidates and reservoirs, temporal and
// spatial, direct light now and indirect later (change-axes.md: "the
// reservoir and the merge rule" belong to Sample reuse, so ReSTIR GI
// changes no line here).
//
// Depends on contracts 4 and 14 alone: what a sample's target is, and the
// numbers it chooses by, are its caller's to pass.
//
// A reservoir in registers (Streaming, below) is contract 14's stored
// reservoir (core/contracts/reservoir.h) with what streaming needs and a
// stored reservoir does not: the running weight sum w_sum, and which input
// the kept sample came from, which the normalization asks (metal/reuse/
// unbiased.metal.h); and the kept sample's target at this surface and its
// visibility test's offset and age, which the stored reservoir keeps.
//
// The operations, each the whole of its step:
//
//   1. Start empty: no sample, w_sum = 0, M = 0.
//   2. Stream a candidate y drawn with density p, at this pixel's surface q
//      whose target is p̂_q (metal/integrator/target.metal.h): w = p̂_q(y)
//      / p(y), w_sum += w, M += 1, and keep y if u w_sum < w, u a number in
//      [0, 1): kept with probability w / w_sum, so after the stream each
//      candidate is the one kept with probability its weight over the
//      total. w = 0 is never kept; a p of 0 is never streamed (the caller
//      has no sample). The caller gives w, not p: the target and the
//      density share factors that cancel in their quotient, which the
//      Integrator computes cancelled (target.metal.h, candidate_weight),
//      so the reservoir never divides two numbers that both vanish at a
//      light's rim. A kept candidate's test is not yet made: its offset
//      and age are 0, here and now, for the test the caller makes next
//      (visibility reuse).
//   3. Stream a reservoir r_i, from surface q_i, with its count already
//      capped where the merge caps it: w = p̂_q(y_i) W_i M_i, w_sum += w,
//      M += M_i, keep y_i as in 2, remembering the input's position in the
//      merge and its test as the caller moved it here (its offset plus the
//      offset it is reused across, its age plus 1 if it is last frame's:
//      contracts/reservoir.h). Its W_i makes p̂_q(y_i) W_i the weight a
//      candidate drawn with density 1 / W_i would have had, and M_i counts
//      the candidates it stands for (Bitterli et al. 2020, Algorithm 4).
//      p̂_q(y_i) for the pixel's own input is its stored target, which is
//      q's this frame; for any other, the caller evaluates it.
//   4. Finish: W = w_sum pi / (p̂_q(y) piSum), pi and piSum the merge's
//      MIS-like normalization as the caller computed it by its mode
//      (unbiased.metal.h); 0 if p̂_q(y) or piSum is 0, or nothing was kept.
//      For candidates alone every candidate was drawn by the same density,
//      so pi = 1 and piSum = M, and W = w_sum / (M p̂_q(y)), as resampled
//      importance sampling gives. The stored reservoir's target is p̂_q(y),
//      its test the kept sample's.
//
// The order is fixed: a merge streams its own reservoir first, then the
// temporal input, then the neighbors in the order they were drawn. Which
// input is kept does not depend on the order in distribution (each is kept
// with probability w_i / w_sum), but which one a given set of numbers keeps
// does, and fixing it makes every frame drawn again draw the same (CDSA.23:
// a mergeable state with a fixed combine order; principle 2). Each choice's
// number is the pass's own, from pixel, frame and purpose
// (metal/sampler/sampler.metal.h; GDSA.3).
//
// The parallel with attention (docs/research/2026-10-10-restir-di-layout.md
// has the table). Streaming resampled importance sampling is sampling from
// a softmax over a stream: fed weights exp(s_i), a reservoir keeps key i
// with probability exp(s_i) / sum_j exp(s_j), softmax(s)_i, without ever
// normalizing, as online softmax never holds the whole row. The draw it
// makes is the distribution GDSA.21's random-key race makes, argmax of
// s_i + Gumbel noise, in one pass, though not the same draw for the same
// numbers. Merging two reservoirs, w_sum = w_a + w_b and keep b's sample
// with probability w_b / w_sum, is FlashAttention's combine of two blocks'
// partial softmax state; a reservoir keeps one sample where attention keeps
// the weighted sum of every value, which is the difference worth stating.
// Held by a GPU test (MEMORY.md: sampling shown on the GPU against the
// distribution it should draw from; tests/gpu/restir_test.cpp): a reservoir
// fed exp(score) for a row of attention scores, its kept key counted over
// many draws, the counts held to softmax by a chi-squared test.
// Scores are kept small there, so exp(score) needs no running maximum, which
// is the one thing attention's online softmax adds for its range.
//
// Cost, per operation: one multiply-add on w_sum, one compare against one
// number, a conditional copy of the 8-byte sample and three words; in
// registers, 28 bytes of state, nothing read or written. What a stream
// costs is its target evaluations (the caller's Frame's) and the bytes of
// the reservoirs it reads, which each pass counts.

#include <metal_stdlib>

#include "core/contracts/light_sample.h"
#include "core/contracts/reservoir.h"

namespace serenity {
namespace shaders {

// Which input of a merge a sample came from: the pixel's own reservoir or
// candidates, the temporal input, or neighbor n (n from 0).
constant constexpr uint from_own = 0xffffffffu;
constant constexpr uint from_temporal = 0xfffffffeu;

// A reservoir being streamed into: steps 1 to 3.
struct Streaming {
    serenity::contracts::LightPoint sample;  // y; holding no light when empty
    float weight_sum;                        // w_sum
    uint count;                              // M
    float target;                            // p̂_q(y), for finish() and the stored reservoir
    uint tested;                             // y's test, as contracts/reservoir.h packs it
    uint source;                             // from_own, from_temporal, or a neighbor's index
};

// A candidate, by name, so its two numbers cannot be passed in each
// other's places (I.24).
struct Candidate {
    serenity::contracts::LightPoint sample;  // y
    float target;                            // p̂_q(y)
    float weight;                            // p̂_q(y) / p(y), finite and at least 0 (step 2)
};

// A reservoir streamed into a merge, as the merge weighs it.
struct MergeInput {
    serenity::contracts::Reservoir reservoir;  // r_i, its `tested` moved to this pixel and frame
    uint count;                                // M_i, capped where the merge caps it
    float target;                              // p̂_q(y_i), at the merging surface: the stored one for the own input
    uint source;                               // from_own, from_temporal, or a neighbor's index
};

// Step 1.
inline Streaming empty_reservoir() {
    return Streaming{serenity::contracts::LightPoint{0u, 0u}, 0.0f, 0u, 0.0f, 0u, from_own};
}

// Step 2: `candidate`, kept or not by `u`; whether it was kept, so the
// caller can hold beside the stream what it needs of the survivor (the
// light's answer, for visibility reuse's ray) without asking for it again.
// An in-out parameter, the state streamed into (F.17); a weight of 0 adds
// to M and is never kept.
inline bool stream_candidate(thread Streaming& reservoir, Candidate candidate, float u) {
    reservoir.weight_sum += candidate.weight;
    reservoir.count += 1u;
    if (!(candidate.weight > 0.0f && u * reservoir.weight_sum < candidate.weight)) {
        return false;
    }
    reservoir.sample = candidate.sample;
    reservoir.target = candidate.target;
    reservoir.tested = 0u;  // here and now: the test is the caller's next (visibility reuse)
    reservoir.source = from_own;
    return true;
}

// Step 3: `input`, kept or not by `u`. Declared, and defined with the
// merges that call it (temporal.metal.h, spatial.metal.h): restir_temporal
// and restir_spatial, which the Metal backend refuses by name until then
// (metal/frame/renderer.h).
inline void stream_reservoir(thread Streaming& reservoir, MergeInput input, float u);

// Step 4's normalization, by name, so its two numbers cannot be passed in
// each other's places (I.24): pi and piSum (unbiased.metal.h); {1, M} for
// candidates alone.
struct Normalization {
    float pi;      // the survivor's own input's domain target
    float pi_sum;  // the sum over the inputs of count times domain target
};

// Step 4: the stored reservoir, normalized as the caller computed. Its
// count is M in contract 14's 24 bits: every M the settings allow fits
// (contracts/reservoir.h), so the mask drops nothing.
inline serenity::contracts::Reservoir finish(Streaming reservoir, Normalization normalization) {
    const bool kept = serenity::contracts::holds_light(reservoir.sample) && reservoir.target > 0.0f &&
                      normalization.pi_sum > 0.0f;
    const float weight =
        kept ? reservoir.weight_sum * normalization.pi / (reservoir.target * normalization.pi_sum) : 0.0f;
    return serenity::contracts::Reservoir{reservoir.sample, reservoir.count & serenity::contracts::count_mask,
                                          reservoir.tested, kept ? reservoir.target : 0.0f, weight};
}

// `reservoir` as reused `offset` pixels from where it was stored (the
// reused pixel minus the reusing one, so a test's offset is always where
// it was made minus the pixel holding it) and `frames` later: its test's
// offset plus `offset`, each coordinate clamped to [-127, 127], and its age
// plus `frames`, saturating at 255 (contracts/reservoir.h, tested). What
// the merges stream, so a survivor arrives with its test placed relative
// to the pixel that keeps it. Declared, and defined with the merges, as
// stream_reservoir() is.
inline serenity::contracts::Reservoir reused_from(serenity::contracts::Reservoir reservoir, int2 offset, uint frames);

}  // namespace shaders
}  // namespace serenity
