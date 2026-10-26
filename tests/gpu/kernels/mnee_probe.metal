// Runs manifold next event estimation's solve (metal/integrator/
// mnee.metal.h, newton() and transfer(), or walk() in the MNEE pass's
// rounds) for tests/gpu/mnee_test.cpp, on one
// glass sphere between a receiver's point and a light's, seeded as
// seed_segment() seeds it, by the straight segment's two crossings; and
// mnee::chain_after() alone, on bounces whose shape may be no_primitive.
// Bindings: the output at 0, the queries at 1 (mnee_probe.h), their count
// at 2; for mnee_chain_after, the shape records at 3.

#include <metal_stdlib>

#include "metal/device/layout.metal.h"
#include "metal/integrator/mnee.metal.h"
#include "mnee_probe.h"

using namespace serenity::shaders;
using serenity::tests::ChainQuery;
using serenity::tests::MneeChain;
using serenity::tests::MneeQuery;

kernel void mnee_solve(device MneeChain* out [[buffer(0)]],
                       constant MneeQuery* queries [[buffer(1)]],
                       constant uint& count [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= count) {
    return;
  }
  const MneeQuery q = queries[i];
  const float3 x0 = to_float3(q.receiver);
  const float3 y = to_float3(q.light);
  const mnee::Caster sphere{to_float3(q.center), q.radius, q.ior, 0u, 0u,
                            float3(0.0f)};
  const float3 d = metal::normalize(y - x0);
  const mnee::Roots roots = mnee::sphere_roots(x0, d, sphere);
  MneeChain chain{};
  if (!roots.found) {
    out[i] = chain;
    return;
  }
  mnee::Caster casters[mnee::max_interfaces] = {sphere, sphere, sphere, sphere};
  float3 p[mnee::max_interfaces] = {x0 + roots.near * d, x0 + roots.far * d,
                                    float3(0.0f), float3(0.0f)};
  mnee::set_references(2u, p, casters);
  mnee::Blocks blocks[mnee::max_interfaces];
  bool found = false;
  if (q.in_rounds == 0u) {
    found = mnee::newton(2u, p, casters, x0, to_float3(q.facing), y, blocks);
  } else {
    // As the MNEE pass takes it: a round at a time, kept between rounds.
    serenity::contracts::MneeWalk kept =
        mnee::kept_walk(i, mnee::begun_walk(2u, p));
    mnee::Walked walked = mnee::Walked::going;
    for (uint round = 0u; round < serenity::contracts::mnee_rounds &&
                          walked == mnee::Walked::going;
         ++round) {
      mnee::Walk w = mnee::resumed_walk(kept);
      walked = mnee::walk(2u, w, casters, x0, to_float3(q.facing), y, blocks,
                          serenity::contracts::mnee_round_iterations);
      kept = mnee::kept_walk(i, w);
    }
    found = walked == mnee::Walked::converged;
    p[0] = to_float3(kept.p[0]);
    p[1] = to_float3(kept.p[1]);
  }
  chain.first = to_packed(p[0]);
  chain.second = to_packed(p[1]);
  chain.found = found ? 1u : 0u;

  if (found) {
    chain.geometry = mnee::geometry_term(2u, p, casters, blocks, x0, y,
                                         to_float3(q.light_normal));
  }
  out[i] = chain;
}

kernel void mnee_chain_after(device uint* out [[buffer(0)]],
                             constant ChainQuery* queries [[buffer(1)]],
                             constant uint& count [[buffer(2)]],
                             constant serenity::shapes::ShapeRecord* records
                             [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (i >= count) {
    return;
  }
  const ChainQuery q = queries[i];
  out[i] = mnee::chain_after(q.chain, q.lobe, records, q.primitive);
}
