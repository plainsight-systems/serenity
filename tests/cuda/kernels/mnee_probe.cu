// Runs manifold next event estimation's solve (cuda/integrator/mnee.cuh,
// newton() and transfer()) for tests/gpu/mnee_test.cpp, on one glass sphere
// between a receiver's point and a light's, seeded as seed_segment() seeds
// it, by the straight segment's two crossings; and mnee::chain_after()
// alone, on bounces whose shape may be no_primitive. Bindings: the output at
// 0, the queries at 1 (mnee_probe.h), their count at 2; for
// mnee_chain_after, the shape records at 3.

#include "cuda/device/layout.cuh"
#include "cuda/integrator/mnee.cuh"
#include "cuda/math/vector.cuh"
#include "gpu/kernels/mnee_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::ChainQuery;
using serenity::tests::MneeChain;
using serenity::tests::MneeQuery;

extern "C" __global__ void mnee_solve(MneeChain* out, const MneeQuery* queries,
                                      const uint* count_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *count_at) {
    return;
  }
  const MneeQuery q = queries[i];
  const float3 x0 = to_float3(q.receiver);
  const float3 y = to_float3(q.light);
  const mnee::Caster sphere{to_float3(q.center), q.radius, q.ior, 0u, 0u,
                            make_float3(0.0f)};
  const float3 d = normalize(y - x0);
  const mnee::Roots roots = mnee::sphere_roots(x0, d, sphere);
  MneeChain chain{};
  if (!roots.found) {
    out[i] = chain;
    return;
  }
  mnee::Caster casters[mnee::max_interfaces] = {sphere, sphere, sphere, sphere};
  float3 p[mnee::max_interfaces] = {x0 + roots.near * d, x0 + roots.far * d,
                                    make_float3(0.0f), make_float3(0.0f)};
  mnee::set_references(2u, p, casters);
  mnee::Blocks blocks[mnee::max_interfaces];
  const bool found =
      mnee::newton(2u, p, casters, x0, to_float3(q.facing), y, blocks);
  chain.first = to_packed(p[0]);
  chain.second = to_packed(p[1]);
  chain.found = found ? 1u : 0u;

  if (found) {
    chain.geometry = mnee::geometry_term(2u, p, casters, blocks, x0, y,
                                         to_float3(q.light_normal));
  }
  out[i] = chain;
}

extern "C" __global__ void mnee_chain_after(
    uint* out, const ChainQuery* queries, const uint* count_at,
    const serenity::shapes::ShapeRecord* records) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= *count_at) {
    return;
  }
  const ChainQuery q = queries[i];
  out[i] = mnee::chain_after(q.chain, q.lobe, records, q.primitive);
}

}  // namespace shaders
}  // namespace serenity
