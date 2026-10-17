#pragma once

// Axis: Sample reuse (the streaming pick).
//
// Resampled importance sampling one sample at a time (Talbot et al. 2005;
// Bitterli et al. 2020, Algorithms 2 and 3): a stream of weighted samples
// goes in, one comes out, chosen with probability its weight over the
// stream's total, and a running total is all that is kept besides it.
// ReSTIR PT's path tree, its NEE candidates and its merges all choose by
// this one line (metal/passes/restir_pt/restir_pt.metal.h,
// metal/reuse/path_merge.metal.h); what is kept, a path or a light sample,
// is the caller's.
//
// The order is fixed by each caller: which input is kept does not depend on
// the order in distribution (each is kept with probability w_i / w_sum),
// but which one a given set of numbers keeps does, and fixing it makes
// every frame drawn again draw the same (CDSA.23: a mergeable state with a
// fixed combine order; principle 2).
//
// The parallel with attention. Streaming resampled importance sampling is
// sampling from a softmax over a stream: fed weights exp(s_i), a reservoir
// keeps key i with probability exp(s_i) / sum_j exp(s_j), softmax(s)_i,
// without ever normalizing, as online softmax never holds the whole row.
// Merging two reservoirs, w_sum = w_a + w_b and keep b's sample with
// probability w_b / w_sum, is FlashAttention's combine of two blocks'
// partial softmax state; a reservoir keeps one sample where attention keeps
// the weighted sum of every value, which is the difference worth stating.
//
// Cost: one add on w_sum and one compare against one number, in registers.

#include <metal_stdlib>

namespace serenity {
namespace shaders {

// The pick (ES.3: one for every stream): `weight` added to the running
// `weight_sum`, and whether the new sample replaces the kept one, with
// probability weight / weight_sum by `u` in [0, 1). A weight of 0 is never
// kept.
inline bool stream_pick(thread float& weight_sum, float weight, float u) {
    weight_sum += weight;
    return weight > 0.0f && u * weight_sum < weight;
}

}  // namespace shaders
}  // namespace serenity
