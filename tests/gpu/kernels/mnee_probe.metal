// Runs manifold next event estimation's solve (metal/integrator/
// mnee.metal.h, newton() and transfer()) for tests/gpu/mnee_test.cpp, on one
// glass sphere between a receiver's point and a light's, seeded as
// seed_segment() seeds it, by the straight segment's two crossings; and
// mnee::chain_after() alone, on bounces whose shape may be no_primitive.
// Bindings: the output at 0, the queries at 1 (mnee_probe.h); for
// mnee_chain_after, the shape records at 2.

#include <metal_stdlib>

#include "metal/device/layout.metal.h"
#include "metal/integrator/mnee.metal.h"
#include "mnee_probe.h"

using namespace serenity::shaders;
using serenity::tests::ChainQuery;
using serenity::tests::MneeChain;
using serenity::tests::MneeQuery;

kernel void mnee_solve(device MneeChain* out [[buffer(0)]], constant MneeQuery* queries [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    const MneeQuery q = queries[i];
    const float3 x0 = to_float3(q.receiver);
    const float3 y = to_float3(q.light);
    const mnee::Caster sphere{to_float3(q.center), q.radius, q.ior, 0u, 0u, float3(0.0f)};
    const float3 d = metal::normalize(y - x0);
    const mnee::Roots roots = mnee::sphere_roots(x0, d, sphere);
    MneeChain chain{};
    if (!roots.found) {
        out[i] = chain;
        return;
    }
    mnee::Caster casters[mnee::max_interfaces] = {sphere, sphere, sphere, sphere};
    float3 p[mnee::max_interfaces] = {x0 + roots.near * d, x0 + roots.far * d, float3(0.0f), float3(0.0f)};
    mnee::set_references(2u, p, casters);
    mnee::Blocks blocks[mnee::max_interfaces];
    const bool found = mnee::newton(2u, p, casters, x0, to_float3(q.facing), y, blocks);
    chain.first = to_packed(p[0]);
    chain.second = to_packed(p[1]);
    chain.found = found ? 1u : 0u;

    if (found) {
        chain.geometry = mnee::geometry_term(2u, p, casters, blocks, x0, y, to_float3(q.light_normal));
    }
    out[i] = chain;
}

kernel void mnee_chain_after(device uint* out [[buffer(0)]], constant ChainQuery* queries [[buffer(1)]],
                             constant serenity::shapes::ShapeRecord* records [[buffer(2)]],
                             uint i [[thread_position_in_grid]]) {
    const ChainQuery q = queries[i];
    out[i] = mnee::chain_after(q.chain, q.lobe, records, q.primitive);
}
