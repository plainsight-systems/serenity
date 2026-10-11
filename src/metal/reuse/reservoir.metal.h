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
// A reservoir in registers (Streaming, below) is contract 14's stored
// reservoir (core/contracts/reservoir.h) with what streaming needs and a
// stored reservoir does not: the running weight sum w_sum, the target at
// the kept sample, and which input it came from, which the unbiased
// normalization asks (metal/reuse/unbiased.metal.h).
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
//      has no sample).
//   3. Stream a reservoir r_i, from surface q_i, with its count already
//      capped where the merge caps it: w = p̂_q(y_i) W_i M_i, w_sum += w,
//      M += M_i, keep y_i as in 2, remembering the input's position in the
//      merge. Its W_i makes p̂_q(y_i) W_i the weight a candidate drawn with
//      density 1 / W_i would have had, and M_i counts the candidates it
//      stands for (Bitterli et al. 2020, Algorithm 4).
//   4. Finish: W = w_sum / (Z p̂_q(y)), Z the count of candidates that could
//      have produced y, as the caller counted it (unbiased.metal.h); 0 if
//      p̂_q(y) or Z is 0, or nothing was kept. For candidates alone every
//      candidate was drawn by the same density, so Z = M, and W = w_sum /
//      (M p̂_q(y)), as resampled importance sampling gives.
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
// Held by a GPU test to be written with the implementation (MEMORY.md:
// sampling shown on the GPU against the distribution it should draw from):
// a reservoir fed exp(score) for a row of attention scores, its kept key
// counted over many draws, the counts held to softmax by a chi-squared test.
// Scores are kept small there, so exp(score) needs no running maximum, which
// is the one thing attention's online softmax adds for its range.
//
// Cost, per operation: one multiply-add on w_sum, one compare against one
// number, a conditional copy of 8 bytes and two words; in registers, nothing
// read or written. What a stream costs is its target evaluations
// (target.metal.h) and the bytes of the reservoirs it reads, which each
// pass counts.

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
    serenity::contracts::LightPoint sample;  // y; light no_light when empty
    float weight_sum;                        // w_sum
    uint count;                              // M
    float target;                            // p̂_q(y), for finish()
    uint source;                             // from_own, from_temporal, or a neighbor's index
};

// A candidate, by name, so its two numbers cannot be passed in each
// other's places (I.24).
struct Candidate {
    serenity::contracts::LightPoint sample;  // y
    float target;                            // p̂_q(y)
    float density;                           // p(y), > 0, in the same measure
};

// A reservoir streamed into a merge, as the merge weighs it.
struct MergeInput {
    serenity::contracts::Reservoir reservoir;  // r_i
    uint count;                                // M_i, capped where the merge caps it
    float target;                              // p̂_q(y_i), at the merging surface
    uint source;                               // from_own, from_temporal, or a neighbor's index
};

// Step 1.
inline Streaming empty_reservoir();

// Step 2: `candidate`, kept or not by `u`.
inline void stream_candidate(thread Streaming& reservoir, Candidate candidate, float u);

// Step 3: `input`, kept or not by `u`.
inline void stream_reservoir(thread Streaming& reservoir, MergeInput input, float u);

// Step 4: the stored reservoir, Z as the caller counted it.
inline serenity::contracts::Reservoir finish(Streaming reservoir, uint z);

}  // namespace shaders
}  // namespace serenity
