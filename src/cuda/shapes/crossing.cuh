#pragma once

// Axis: Shape.
//
// What a shape kind's exact test answers of a ray (shapes.cuh): whether
// the ray crosses its surface within the ray's (min_distance, max_distance),
// and at what parameter t. One result, returned, rather than a bool and an
// out parameter (F.20, F.21).
//
// Translated line for line from Metal's metal/shapes/crossing.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "cuda/math/vector.cuh"

namespace serenity {
namespace shaders {

struct Crossing {
  bool found;
  float t;  // the ray's parameter at the crossing; 0 where none was found
};

__device__ inline Crossing no_crossing() { return Crossing{false, 0.0f}; }

}  // namespace shaders
}  // namespace serenity
