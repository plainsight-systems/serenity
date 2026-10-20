#pragma once

// Axis: Material (the BSDF contract's shader side, contracts/bsdf.h).
//
// The four questions of contract 2, asked of any resolved Bsdf, answered by
// its kind: the one place a shader dispatches on BsdfKind, so an estimator
// names none. Each switch has no default, so a kind added to BsdfKind and
// not here fails Metal's compile (-Werror), and here, where nvcc does not warn
// of an enumerator a switch leaves out, traps (device/layout.cuh,
// unhandled_kind).
//
// Reflecting kinds are two-sided: they reflect on whichever side wo is, the
// shading normal turned to face it. Glass decides entering from leaving by
// the unturned normal.
//
// "No sample" is one value everywhere: BsdfSample{}, its pdf 0 (contract
// 2), every other field 0.
//
// Translated line for line from Metal's metal/materials/bsdf.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include <cuda_fp16.h>

#include "core/contracts/bsdf.h"
#include "cuda/device/layout.cuh"
#include "cuda/materials/coated.cuh"
#include "cuda/materials/conductor.cuh"
#include "cuda/materials/dielectric.cuh"
#include "cuda/materials/rough.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/math/warp.cuh"

namespace serenity {
namespace shaders {

__device__ inline float3 facing(serenity::contracts::Bsdf bsdf, float3 wo) {
  const float3 n = to_float3(bsdf.normal);
  return dot(n, wo) < 0.0f ? -n : n;
}

__device__ inline float3 bsdf_evaluate(serenity::contracts::Bsdf bsdf,
                                       float3 wo, float3 wi) {
  switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
      return make_float3(0.0f);
    case serenity::contracts::BsdfKind::lambert:
      return lambert_evaluate(bsdf, facing(bsdf, wo), wi);
    case serenity::contracts::BsdfKind::conductor:
      return conductor_evaluate(bsdf, facing(bsdf, wo), wo, wi);
    case serenity::contracts::BsdfKind::dielectric:
      return make_float3(0.0f);  // delta lobes only
    case serenity::contracts::BsdfKind::coated:
      return coated_evaluate(bsdf, facing(bsdf, wo), wo, wi);
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return make_float3(0.0f);
}

__device__ inline float bsdf_pdf(serenity::contracts::Bsdf bsdf, float3 wo,
                                 float3 wi) {
  switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
      return 0.0f;
    case serenity::contracts::BsdfKind::lambert:
      return lambert_pdf(facing(bsdf, wo), wi);
    case serenity::contracts::BsdfKind::conductor:
      return conductor_pdf(bsdf, facing(bsdf, wo), wo, wi);
    case serenity::contracts::BsdfKind::dielectric:
      return 0.0f;  // delta lobes only
    case serenity::contracts::BsdfKind::coated:
      return coated_pdf(bsdf, facing(bsdf, wo), wo, wi);
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return 0.0f;
}

// `u` is three numbers in [0, 1): u.x chooses a lobe, u.yz the direction.
__device__ inline serenity::contracts::BsdfSample bsdf_sample(
    serenity::contracts::Bsdf bsdf, float3 wo, float3 u) {
  switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
      return serenity::contracts::BsdfSample{};  // scatters nothing: no sample
    case serenity::contracts::BsdfKind::lambert:
      return lambert_sample(bsdf, facing(bsdf, wo), yz(u));
    case serenity::contracts::BsdfKind::conductor:
      return conductor_sample(bsdf, facing(bsdf, wo), wo, yz(u));
    case serenity::contracts::BsdfKind::dielectric:
      return dielectric_sample(bsdf, to_float3(bsdf.normal), wo, u.x);
    case serenity::contracts::BsdfKind::coated:
      return coated_sample(bsdf, facing(bsdf, wo), wo, u);
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return serenity::contracts::BsdfSample{};
}

// eta_t of a transmission sample drawn for wo: the index of refraction on
// wi's side over wo's, by which the sample's value carries radiance's
// 1 / eta_t^2 (dielectric.cuh). 1 for a kind that does not transmit.
// pbrt-v4's BSDFSample::eta, here a question of the Bsdf and wo because
// contract 2's BsdfSample does not carry it; the path tracer's roulette
// undoes it (integrator/path.cuh, step 7).
__device__ inline float bsdf_eta(serenity::contracts::Bsdf bsdf, float3 wo) {
  switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
    case serenity::contracts::BsdfKind::lambert:
    case serenity::contracts::BsdfKind::conductor:
    case serenity::contracts::BsdfKind::coated:
      return 1.0f;
    case serenity::contracts::BsdfKind::dielectric:
      return transmitted_eta(bsdf.ior, dot(wo, to_float3(bsdf.normal)) > 0.0f);
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return 1.0f;
}

__device__ inline uint bsdf_lobes(serenity::contracts::Bsdf bsdf) {
  switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
      return 0u;
    case serenity::contracts::BsdfKind::lambert:
      return serenity::contracts::lobe_reflection |
             serenity::contracts::lobe_diffuse;
    case serenity::contracts::BsdfKind::conductor:
      return serenity::contracts::lobe_reflection |
             serenity::contracts::lobe_glossy;
    case serenity::contracts::BsdfKind::dielectric:
      return serenity::contracts::lobe_reflection |
             serenity::contracts::lobe_transmission |
             serenity::contracts::lobe_delta;
    case serenity::contracts::BsdfKind::coated:
      return serenity::contracts::lobe_reflection |
             serenity::contracts::lobe_diffuse |
             serenity::contracts::lobe_delta;
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return 0u;
}

// Contract 2's packed form (contracts/bsdf.h states both functions and
// what packing loses): Material's one definition of how a Bsdf is stored in
// ReSTIR PT's G-buffer and MNEE's requests and read back, by every pass
// that writes or reads one. A switch on the kind with no default, as above, so
// a kind added and not packed fails Metal's compile, and here traps.
//
// A half is rounded from a float to nearest, ties to even, by the shading
// language's conversion; a positive value that rounds to 0 is raised to
// half's least, 2^-24. Precondition: each value finite and at most 65504
// in magnitude, which a renderer that packs has checked of the scene
// (contracts/bsdf.h states both, and the tolerance). Cost: pack, one
// encode_unit and four conversions; unpack, one decode_unit and four
// widenings (GDSA.18: decoded where it is loaded, never stored decoded).
// Pure (F.8).

// `value` as a half's bits, a positive value never 0.
inline constexpr uint half_magnitude = 0x7fffu;  // a half's bits but its sign
// the least positive half, 2^-24, a subnormal
inline constexpr uint half_least = 1u;

__device__ inline uint pack_half(float value) {
  const ushort bits = __half_as_ushort(__float2half_rn(value));
  return value > 0.0f && (bits & half_magnitude) == 0u ? half_least
                                                       : uint(bits);
}

__device__ inline float unpack_half(uint bits) {
  return __half2float(
      __ushort_as_half(ushort(bits & serenity::contracts::packed_half_mask)));
}

// The scalar in color_b_scalar's high half: the one of alpha and ior the
// kind reads, 0 for a kind that reads neither (contracts/bsdf.h).
__device__ inline float packed_scalar(serenity::contracts::Bsdf bsdf) {
  switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
    case serenity::contracts::BsdfKind::lambert:
      return 0.0f;
    case serenity::contracts::BsdfKind::conductor:
      return bsdf.alpha;
    case serenity::contracts::BsdfKind::dielectric:
    case serenity::contracts::BsdfKind::coated:
      return bsdf.ior;
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return 0.0f;
}

__device__ inline serenity::contracts::PackedBsdf pack_bsdf(
    serenity::contracts::Bsdf bsdf) {
  const bool escapes = bsdf.kind == serenity::contracts::BsdfKind::coated;
  return serenity::contracts::PackedBsdf{
      encode_unit(to_float3(bsdf.normal)),
      pack_half(bsdf.color.x) |
          (pack_half(bsdf.color.y) << serenity::contracts::packed_high_shift),
      pack_half(bsdf.color.z) | (pack_half(packed_scalar(bsdf))
                                 << serenity::contracts::packed_high_shift),
      (escapes ? pack_half(bsdf.escape) : 0u) |
          (uint(bsdf.kind) << serenity::contracts::packed_kind_shift),
  };
}

using serenity::contracts::packed_kind;  // contracts/bsdf.h

__device__ inline serenity::contracts::Bsdf unpack_bsdf(
    serenity::contracts::PackedBsdf packed) {
  serenity::contracts::Bsdf bsdf{};
  bsdf.normal = to_packed(decode_unit(packed.normal));
  bsdf.kind = packed_kind(packed);
  bsdf.color = serenity::contracts::Float3{
      unpack_half(packed.color_rg),
      unpack_half(packed.color_rg >> serenity::contracts::packed_high_shift),
      unpack_half(packed.color_b_scalar)};
  const float scalar = unpack_half(packed.color_b_scalar >>
                                   serenity::contracts::packed_high_shift);
  switch (bsdf.kind) {
    case serenity::contracts::BsdfKind::none:
    case serenity::contracts::BsdfKind::lambert:
      return bsdf;
    case serenity::contracts::BsdfKind::conductor:
      bsdf.alpha = scalar;
      return bsdf;
    case serenity::contracts::BsdfKind::dielectric:
      bsdf.ior = scalar;
      return bsdf;
    case serenity::contracts::BsdfKind::coated:
      bsdf.ior = scalar;
      bsdf.escape = unpack_half(packed.escape_kind);
      return bsdf;
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return bsdf;
}

}  // namespace shaders
}  // namespace serenity
