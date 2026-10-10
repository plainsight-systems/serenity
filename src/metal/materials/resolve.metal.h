#pragma once

// Axis: Material (resolving, contracts/bsdf.h).
//
// A material resolved at a surface: its kind's data read, its textures
// evaluated at the point, into the Bsdf every later question reads. The one
// place a shader dispatches on MaterialKind; the switch has no default, so
// a kind added to MaterialKind and not here fails to compile (-Werror).
//
//   rough       lambert: color from its texture or constant
//   conductor   conductor: f0, alpha = roughness^2, at least 1e-3 (a GGX
//               with alpha 0 has no density to sample)
//   dielectric  dielectric: ior
//   coated      coated: color from its texture or constant, the coat's
//               ior, and its internal reflectance (materials/coated.h)
//   emissive    none: it scatters nothing; its light is the emitter's
//
// A texture is read at the surface's point in the world and in the shape's
// own coordinates (textures/texture.h), both from the surface interaction
// (contract 1), so a core's swirl moves with its core.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "core/materials/coated.h"
#include "core/materials/conductor.h"
#include "core/materials/dielectric.h"
#include "core/materials/material.h"
#include "core/materials/rough.h"
#include "metal/device/layout.metal.h"
#include "metal/materials/rough.metal.h"
#include "metal/textures/textures.metal.h"

namespace serenity {
namespace shaders {

// Every material, as a shader reads it: the records and one array per kind.
struct Materials {
    device const serenity::materials::MaterialRecord* records;
    device const serenity::materials::RoughData* rough;
    device const serenity::materials::DielectricData* dielectrics;
    device const serenity::materials::ConductorData* conductors;
    device const serenity::materials::CoatedData* coated;
};

inline serenity::contracts::Bsdf resolve_bsdf(Materials materials, Textures textures,
                                              serenity::contracts::SurfaceInteraction surface) {
    const serenity::materials::MaterialRecord record = materials.records[surface.material];
    serenity::contracts::Bsdf bsdf;
    bsdf.normal = surface.shading_normal;
    bsdf.color = to_packed(float3(1.0f));
    bsdf.alpha = 0.0f;
    bsdf.ior = 0.0f;
    bsdf.escape = 0.0f;
    bsdf.padding[0] = bsdf.padding[1] = 0u;
    const float3 world = to_float3(surface.position);
    const float3 object = to_float3(surface.object_position);
    switch (record.kind) {
    case serenity::materials::MaterialKind::rough:
        bsdf.kind = serenity::contracts::BsdfKind::lambert;
        bsdf.color = to_packed(rough_color(materials.rough[record.index], textures, world, object));
        break;
    case serenity::materials::MaterialKind::conductor: {
        const serenity::materials::ConductorData conductor = materials.conductors[record.index];
        bsdf.kind = serenity::contracts::BsdfKind::conductor;
        bsdf.color = conductor.f0;
        bsdf.alpha = metal::max(conductor.roughness * conductor.roughness, 1e-3f);
        break;
    }
    case serenity::materials::MaterialKind::dielectric:
        bsdf.kind = serenity::contracts::BsdfKind::dielectric;
        bsdf.ior = materials.dielectrics[record.index].ior;
        break;
    case serenity::materials::MaterialKind::emissive:
        bsdf.kind = serenity::contracts::BsdfKind::none;
        break;
    case serenity::materials::MaterialKind::coated: {
        const serenity::materials::CoatedData coat = materials.coated[record.index];
        bsdf.kind = serenity::contracts::BsdfKind::coated;
        serenity::materials::RoughData base;
        base.color = coat.color;
        base.texture = coat.texture;
        bsdf.color = to_packed(rough_color(base, textures, world, object));
        bsdf.ior = coat.ior;
        bsdf.escape = coat.escape;
        break;
    }
    }
    return bsdf;
}

}  // namespace shaders
}  // namespace serenity
