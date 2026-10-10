// Runs the BSDF contract's shader side (metal/materials/bsdf.metal.h,
// resolve.metal.h) for tests/gpu/bsdf_test.cpp. Bindings: the output at 0,
// the inputs after it in the order each kernel lists them (probes.h).

#include <metal_stdlib>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "metal/materials/bsdf.metal.h"
#include "metal/materials/resolve.metal.h"
#include "metal/math/hash.metal.h"
#include "metal/scene/scene_block.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::contracts::Bsdf;
using serenity::tests::BsdfCell;
using serenity::tests::BsdfDraw;
using serenity::tests::BsdfQuery;

// Draw i of query.count, from numbers hashed from i. Inputs: the Bsdf, the
// query.
kernel void bsdf_samples(device BsdfDraw* out [[buffer(0)]],
                         constant Bsdf& bsdf [[buffer(1)]],
                         constant BsdfQuery& query [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= query.count) {
        return;
    }
    const float3 wo = to_float3(query.wo);
    const uint h = pcg_hash(i);
    const float3 u = float3(unit_float(h), unit_float(pcg_hash(h)), unit_float(pcg_hash(h ^ 0x9e3779b9u)));
    const serenity::contracts::BsdfSample s = bsdf_sample(bsdf, wo, u);
    const float3 wi = to_float3(s.direction);
    out[i] = BsdfDraw{s.direction,
                      s.pdf,
                      s.value,
                      s.lobe,
                      to_packed(s.pdf > 0.0f ? bsdf_evaluate(bsdf, wo, wi) : float3(0.0f)),
                      s.pdf > 0.0f ? bsdf_pdf(bsdf, wo, wi) : 0.0f};
}

// evaluate() and pdf() over the sphere, in query.count x query.count cells
// of equal solid angle: z = 1 - 2 (row + 0.5) / count, azimuth
// 2 pi (column + 0.5) / count. Inputs: the Bsdf, the query.
kernel void bsdf_density(device BsdfCell* out [[buffer(0)]],
                         constant Bsdf& bsdf [[buffer(1)]],
                         constant BsdfQuery& query [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    const uint res = query.count;
    if (i >= res * res) {
        return;
    }
    const float z = 1.0f - 2.0f * (float(i / res) + 0.5f) / float(res);
    const float phi = 2.0f * M_PI_F * (float(i % res) + 0.5f) / float(res);
    const float r = metal::sqrt(metal::max(0.0f, 1.0f - z * z));
    const float3 wi = float3(r * metal::cos(phi), r * metal::sin(phi), z);
    const float3 wo = to_float3(query.wo);
    out[i] = BsdfCell{to_packed(bsdf_evaluate(bsdf, wo, wi)), bsdf_pdf(bsdf, wo, wi)};
}

// resolve() for each of `count` surfaces, the materials and textures read
// through the scene's block as the passes read them (metal/scene/
// scene_block.metal.h). Inputs: the surfaces, the count, the block.
kernel void bsdf_resolve(device Bsdf* out [[buffer(0)]],
                         device const serenity::contracts::SurfaceInteraction* surfaces [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         constant serenity::gpu::SceneBlock& block [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= count) {
        return;
    }
    out[i] = resolve_bsdf(Materials{block.materials, block.roughs, block.dielectrics, block.conductors, block.coateds},
                          Textures{block.textures, block.checkers, block.woods, block.swirls}, surfaces[i]);
}

// lobes() of each of `count` Bsdfs. Inputs: the Bsdfs, the count.
kernel void bsdf_lobe_bits(device uint* out [[buffer(0)]],
                           device const Bsdf* bsdfs [[buffer(1)]],
                           constant uint& count [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = bsdf_lobes(bsdfs[i]);
    }
}

// bsdf_eta() of the Bsdf for query.wo, in out[0]. Inputs: the Bsdf, the
// query.
kernel void bsdf_eta_of(device float* out [[buffer(0)]],
                        constant Bsdf& bsdf [[buffer(1)]],
                        constant BsdfQuery& query [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i == 0) {
        out[0] = bsdf_eta(bsdf, to_float3(query.wo));
    }
}
