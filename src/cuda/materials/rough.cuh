#pragma once

// Axis: Material (shader half of materials/rough.h).
//
// A rough surface: its color at a point, its texture's through the texture
// reference or its constant color; and its BSDF (contracts/bsdf.h), Lambert's,
// albedo / pi on the side wo is on, sampled by the cosine (Malley's method),
// pdf = |cos wi| / pi. One lobe: reflection, diffuse.
//
// Translated line for line from Metal's metal/materials/rough.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/bsdf.h"
#include "core/materials/rough.h"
#include "cuda/device/layout.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/math/warp.cuh"
#include "cuda/textures/textures.cuh"

namespace serenity {
namespace shaders {

// `world` and `object`: the point, in the world and in its shape's own
// coordinates (evaluate_texture).
__device__ inline float3 rough_color(serenity::materials::RoughData rough,
                                     Textures textures, float3 world,
                                     float3 object) {
  if (rough.texture.index == serenity::contracts::no_texture) {
    return to_float3(rough.color);
  }
  return evaluate_texture(textures, rough.texture, world, object);
}

// `n` is the normal turned to wo's side; wi on the other side scatters
// nothing.
__device__ inline float3 lambert_evaluate(serenity::contracts::Bsdf bsdf,
                                          float3 n, float3 wi) {
  return dot(wi, n) > 0.0f ? to_float3(bsdf.color) * inverse_pi
                           : make_float3(0.0f);
}

__device__ inline float lambert_pdf(float3 n, float3 wi) {
  return max(0.0f, dot(wi, n)) * inverse_pi;
}

__device__ inline serenity::contracts::BsdfSample lambert_sample(
    serenity::contracts::Bsdf bsdf, float3 n, float2 u) {
  const float3 wi = cosine_direction(n, u);
  return serenity::contracts::BsdfSample{
      to_packed(wi), lambert_pdf(n, wi),
      to_packed(lambert_evaluate(bsdf, n, wi)),
      serenity::contracts::lobe_reflection | serenity::contracts::lobe_diffuse};
}

}  // namespace shaders
}  // namespace serenity
