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
//   emissive    none: it scatters nothing; its light is the emitter's

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
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
};

inline serenity::contracts::Bsdf resolve_bsdf(Materials materials, Textures textures,
                                              serenity::contracts::SurfaceInteraction surface) {
    const serenity::materials::MaterialRecord record = materials.records[surface.material];
    serenity::contracts::Bsdf bsdf;
    bsdf.normal = surface.shading_normal;
    bsdf.color = to_packed(float3(1.0f));
    bsdf.alpha = 0.0f;
    bsdf.ior = 0.0f;
    bsdf.padding[0] = bsdf.padding[1] = bsdf.padding[2] = 0u;
    switch (record.kind) {
    case serenity::materials::MaterialKind::rough:
        bsdf.kind = serenity::contracts::BsdfKind::lambert;
        bsdf.color = to_packed(rough_color(materials.rough[record.index], textures, to_float3(surface.position)));
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
    }
    return bsdf;
}

}  // namespace shaders
}  // namespace serenity
