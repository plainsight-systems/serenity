#pragma once

// Axis: Material (resolving, contracts/bsdf.h).
//
// A material resolved at a surface: its kind's data read, its textures
// evaluated at the point, into the Bsdf every later question reads. The one
// place a shader dispatches on MaterialKind; the switch has no default, so
// a kind added to MaterialKind and not here fails Metal's compile (-Werror),
// and here, where nvcc does not warn of an enumerator a switch leaves out,
// traps (device/layout.cuh, unhandled_kind).
//
//   rough       lambert: color from its texture or constant
//   conductor   conductor: f0, alpha = roughness^2, no less than least_alpha
//               (a GGX with alpha 0 has no density to sample)
//   dielectric  dielectric: ior
//   coated      coated: color from its texture or constant, the coat's
//               ior, and its internal reflectance (materials/coated.h)
//   emissive    none: it scatters nothing; its light is the emitter's
//
// A texture is read at the surface's point in the world and in the shape's
// own coordinates (textures/texture.h), both from the surface interaction
// (contract 1), so a core's swirl moves with its core.
//
// Translated line for line from Metal's metal/materials/resolve.metal.h
// (docs/research/2026-10-19-cuda-backend.md, "Phase 3's design"): its
// prose is Metal's, and a cost or a measurement it gives is the M3 Max's.

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "core/materials/coated.h"
#include "core/materials/conductor.h"
#include "core/materials/dielectric.h"
#include "core/materials/material.h"
#include "core/materials/rough.h"
#include "cuda/device/layout.cuh"
#include "cuda/materials/rough.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/textures/textures.cuh"

namespace serenity {
namespace shaders {

// The least GGX alpha a conductor resolves to: a GGX with alpha 0 has no
// density to sample.
inline constexpr float least_alpha = 1e-3f;

// Every material, as a shader reads it: the records and one array per kind.
struct Materials {
  const serenity::materials::MaterialRecord* records;
  const serenity::materials::RoughData* roughs;
  const serenity::materials::DielectricData* dielectrics;
  const serenity::materials::ConductorData* conductors;
  const serenity::materials::CoatedData* coateds;
};

__device__ inline serenity::contracts::Bsdf resolve_bsdf(
    Materials materials, Textures textures,
    serenity::contracts::SurfaceInteraction surface) {
  const serenity::materials::MaterialRecord record =
      materials.records[surface.material];
  // Every field set before the switch (ES.20), as Metal's, where a record
  // whose kind is no enumerator resolves to none, which scatters nothing;
  // here such a record traps after the switch (layout.cuh).
  serenity::contracts::Bsdf bsdf{};
  bsdf.normal = surface.shading_normal;
  bsdf.kind = serenity::contracts::BsdfKind::none;
  bsdf.color = to_packed(make_float3(1.0f));
  const float3 world = to_float3(surface.position);
  const float3 object = to_float3(surface.object_position);
  switch (record.kind) {
    case serenity::materials::MaterialKind::rough:
      bsdf.kind = serenity::contracts::BsdfKind::lambert;
      bsdf.color = to_packed(
          rough_color(materials.roughs[record.index], textures, world, object));
      return bsdf;
    case serenity::materials::MaterialKind::conductor: {
      const serenity::materials::ConductorData conductor =
          materials.conductors[record.index];
      bsdf.kind = serenity::contracts::BsdfKind::conductor;
      bsdf.color = conductor.f0;
      bsdf.alpha = max(conductor.roughness * conductor.roughness, least_alpha);
      return bsdf;
    }
    case serenity::materials::MaterialKind::dielectric:
      bsdf.kind = serenity::contracts::BsdfKind::dielectric;
      bsdf.ior = materials.dielectrics[record.index].ior;
      return bsdf;
    case serenity::materials::MaterialKind::emissive:
      bsdf.kind = serenity::contracts::BsdfKind::none;
      return bsdf;
    case serenity::materials::MaterialKind::coated: {
      const serenity::materials::CoatedData coat =
          materials.coateds[record.index];
      bsdf.kind = serenity::contracts::BsdfKind::coated;
      serenity::materials::RoughData base{};
      base.color = coat.color;
      base.texture = coat.texture;
      bsdf.color = to_packed(rough_color(base, textures, world, object));
      bsdf.ior = coat.ior;
      bsdf.escape = coat.escape;
      return bsdf;
    }
  }
  unhandled_kind();  // a kind no case handles (layout.cuh)
  return bsdf;
}

}  // namespace shaders
}  // namespace serenity
