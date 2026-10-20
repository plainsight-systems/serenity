// Runs the BSDF contract's shader side (cuda/materials/bsdf.cuh,
// metal/kernels/resolve.metal.h) for tests/gpu/bsdf_test.cpp. Bindings: the
// output at 0, the inputs after it in the order each kernel lists them
// (bsdf_probe.h).

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "core/scene/scene_block.h"
#include "cuda/materials/bsdf.cuh"
#include "cuda/materials/resolve.cuh"
#include "cuda/math/hash.cuh"
#include "cuda/math/vector.cuh"
#include "gpu/kernels/bsdf_probe.h"

namespace serenity {
namespace shaders {

using serenity::contracts::Bsdf;
using serenity::tests::BsdfCell;
using serenity::tests::BsdfDraw;
using serenity::tests::BsdfQuery;

// Draw i of query.count, from numbers hashed from i. Inputs: the Bsdf, the
// query.
extern "C" __global__ void bsdf_samples(BsdfDraw* out, const Bsdf* bsdf_at,
                                        const BsdfQuery* query_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const Bsdf& bsdf = *bsdf_at;
  const BsdfQuery& query = *query_at;
  if (i >= query.count) {
    return;
  }
  const float3 wo = to_float3(query.wo);
  const uint h = pcg_hash(i);
  const float3 u = make_float3(unit_float(h), unit_float(pcg_hash(h)),
                               unit_float(pcg_hash(h ^ 0x9e3779b9u)));
  const serenity::contracts::BsdfSample s = bsdf_sample(bsdf, wo, u);
  const float3 wi = to_float3(s.direction);
  out[i] = BsdfDraw{
      s.direction,
      s.pdf,
      s.value,
      s.lobe,
      to_packed(s.pdf > 0.0f ? bsdf_evaluate(bsdf, wo, wi) : make_float3(0.0f)),
      s.pdf > 0.0f ? bsdf_pdf(bsdf, wo, wi) : 0.0f};
}

// evaluate() and pdf() over the sphere, in query.count x query.count cells
// of equal solid angle: z = 1 - 2 (row + 0.5) / count, azimuth
// 2 pi (column + 0.5) / count. Inputs: the Bsdf, the query.
extern "C" __global__ void bsdf_density(BsdfCell* out, const Bsdf* bsdf_at,
                                        const BsdfQuery* query_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const Bsdf& bsdf = *bsdf_at;
  const BsdfQuery& query = *query_at;
  const uint res = query.count;
  if (i >= res * res) {
    return;
  }
  const float z = 1.0f - 2.0f * (float(i / res) + 0.5f) / float(res);
  const float phi = 2.0f * pi * (float(i % res) + 0.5f) / float(res);
  const float r = sqrt(max(0.0f, 1.0f - z * z));
  const float3 wi = make_float3(r * cos(phi), r * sin(phi), z);
  const float3 wo = to_float3(query.wo);
  out[i] =
      BsdfCell{to_packed(bsdf_evaluate(bsdf, wo, wi)), bsdf_pdf(bsdf, wo, wi)};
}

// resolve() for each of `count` surfaces, the materials and textures read
// through the scene's block as the passes read them (metal/scene/
// metal/kernels/scene_block.metal.h). Inputs: the surfaces, the count, the
// block.
extern "C" __global__ void bsdf_resolve(
    Bsdf* out, const serenity::contracts::SurfaceInteraction* surfaces,
    const uint* count_at, const serenity::gpu::SceneBlock* block_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  const serenity::gpu::SceneBlock& block = *block_at;
  if (i >= count) {
    return;
  }
  out[i] = resolve_bsdf(
      Materials{block.materials, block.roughs, block.dielectrics,
                block.conductors, block.coateds},
      Textures{block.textures, block.checkers, block.woods, block.swirls},
      surfaces[i]);
}

// lobes() of each of `count` Bsdfs. Inputs: the Bsdfs, the count.
extern "C" __global__ void bsdf_lobe_bits(uint* out, const Bsdf* bsdfs,
                                          const uint* count_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  if (i < count) {
    out[i] = bsdf_lobes(bsdfs[i]);
  }
}

// bsdf_eta() of the Bsdf for query.wo, in out[0]. Inputs: the Bsdf, the
// query.
extern "C" __global__ void bsdf_eta_of(float* out, const Bsdf* bsdf_at,
                                       const BsdfQuery* query_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const Bsdf& bsdf = *bsdf_at;
  const BsdfQuery& query = *query_at;
  if (i == 0) {
    out[0] = bsdf_eta(bsdf, to_float3(query.wo));
  }
}

}  // namespace shaders
}  // namespace serenity
