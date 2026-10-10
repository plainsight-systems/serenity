// Runs the BSDF contract's shader side (metal/materials/bsdf.metal.h,
// resolve.metal.h) for tests/gpu/bsdf_test.cpp.

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/sampler/sampler.metal.h"
#include "bsdf_probe.h"

using namespace serenity::shaders;
using serenity::contracts::Bsdf;

using serenity::tests::Probe;

// Sample i of `count`, from numbers hashed from i.
kernel void bsdf_samples(device Probe* out [[buffer(0)]],
                         constant Bsdf& bsdf [[buffer(1)]],
                         constant float4& wo_count [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= uint(wo_count.w)) {
        return;
    }
    const float3 wo = wo_count.xyz;
    const uint h = pcg_hash(i);
    const float3 u = float3(unit_float(h), unit_float(pcg_hash(h)), unit_float(pcg_hash(h ^ 0x9e3779b9u)));
    const serenity::contracts::BsdfSample s = bsdf_sample(bsdf, wo, u);
    const float3 wi = to_float3(s.direction);
    Probe p;
    p.direction = s.direction;
    p.pdf = s.pdf;
    p.value = s.value;
    p.lobe = s.lobe;
    p.evaluated = to_packed(s.pdf > 0.0f ? bsdf_evaluate(bsdf, wo, wi) : float3(0.0f));
    p.evaluated_pdf = s.pdf > 0.0f ? bsdf_pdf(bsdf, wo, wi) : 0.0f;
    out[i] = p;
}

// evaluate and pdf over the sphere, in res x res cells of equal solid angle:
// z = 1 - 2 (row + 0.5) / res, azimuth 2 pi (column + 0.5) / res.
kernel void bsdf_density(device float4* out [[buffer(0)]],
                         constant Bsdf& bsdf [[buffer(1)]],
                         constant float4& wo_res [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    const uint res = uint(wo_res.w);
    if (i >= res * res) {
        return;
    }
    const float z = 1.0f - 2.0f * (float(i / res) + 0.5f) / float(res);
    const float phi = 2.0f * M_PI_F * (float(i % res) + 0.5f) / float(res);
    const float r = metal::sqrt(metal::max(0.0f, 1.0f - z * z));
    const float3 wi = float3(r * metal::cos(phi), r * metal::sin(phi), z);
    out[i] = float4(bsdf_evaluate(bsdf, wo_res.xyz, wi), bsdf_pdf(bsdf, wo_res.xyz, wi));
}

// resolve() for each of `count` surfaces.
kernel void bsdf_resolve(device Bsdf* out [[buffer(0)]],
                         device const serenity::contracts::SurfaceInteraction* surfaces [[buffer(1)]],
                         constant serenity::materials::MaterialRecord* records [[buffer(2)]],
                         constant serenity::materials::RoughData* rough [[buffer(3)]],
                         constant serenity::materials::DielectricData* dielectrics [[buffer(4)]],
                         constant serenity::materials::ConductorData* conductors [[buffer(5)]],
                         constant serenity::textures::TextureRecord* texture_records [[buffer(6)]],
                         constant serenity::textures::CheckerData* checkers [[buffer(7)]],
                         constant uint& count [[buffer(8)]],
                         constant serenity::materials::CoatedData* coated [[buffer(9)]],
                         constant serenity::textures::WoodData* woods [[buffer(10)]],
                         constant serenity::textures::SwirlData* swirls [[buffer(11)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    out[i] = resolve_bsdf(Materials{records, rough, dielectrics, conductors, coated},
                          Textures{texture_records, checkers, woods, swirls}, surfaces[i]);
}

// lobes() of each of `count` Bsdfs.
kernel void bsdf_lobe_bits(device uint* out [[buffer(0)]],
                           device const Bsdf* bsdfs [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = bsdf_lobes(bsdfs[i]);
    }
}
