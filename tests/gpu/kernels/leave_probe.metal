// Runs the tracing loop's step 1 and a ray leaving the surface it found
// (metal/integrator/path.metal.h, step 1; integrator/surface.metal.h,
// leave()) for tests/gpu/leave_test.cpp: whether a ray leaving a surface
// hits that surface again. Thread i takes query i / leave_directions: it
// traces the query's ray, finds the surface interaction at its hit as the
// path tracer does (origin + t d, shapes/shapes.metal.h), and traces one
// ray leaving it by leave(), direction i % leave_directions: the outward
// side of the geometric normal for even directions, the inward side for
// odd, at a cosine to the normal from 1 down to grazing, about it at an
// angle hashed from i. Bindings: the output at 0, then the queries, their
// count, the structure, and the shapes' records, transforms and boxes
// (leave_probe.h).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/transform.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "leave_probe.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/device/layout.metal.h"
#include "metal/integrator/surface.metal.h"
#include "metal/math/hash.metal.h"
#include "metal/math/warp.metal.h"
#include "metal/shapes/shapes.metal.h"

using namespace serenity::shaders;
using serenity::tests::LeaveQuery;
using serenity::tests::LeaveResult;

// The cosines to the normal the leaving rays take, head on to grazing.
constant float leave_cosines[serenity::tests::leave_cosine_count] = {1.0f,  0.5f,   0.1f, 0.03f,
                                                                    0.01f, 3e-3f, 1e-3f, 3e-4f};

kernel void leave_probe(device LeaveResult* out [[buffer(0)]],
                        device const LeaveQuery* queries [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        serenity::shaders::SceneStructure structure [[buffer(3)]],
                        constant serenity::shapes::ShapeRecord* records [[buffer(4)]],
                        constant serenity::contracts::Transform* transforms [[buffer(5)]],
                        constant serenity::shapes::BoxData* boxes [[buffer(6)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= count * serenity::tests::leave_directions) {
        return;
    }
    const Shapes shapes{records, transforms, boxes};
    const LeaveQuery query = queries[i / serenity::tests::leave_directions];
    const uint j = i % serenity::tests::leave_directions;
    const metal::raytracing::ray aimed(to_float3(query.origin), to_float3(query.direction), 0.0f, unbounded);
    const Hit found = trace(structure, shapes, aimed, serenity::contracts::RayMask::everything);
    LeaveResult result{};
    if (!found.found || found.primitive != query.shape) {
        out[i] = result;
        return;
    }
    result.found = 1u;
    const serenity::contracts::SurfaceInteraction surface =
        surface_interaction(shapes, found.primitive, aimed.origin + found.t * aimed.direction, aimed.direction);

    const float3 n = to_float3(surface.geometric_normal);
    const float cosine = leave_cosines[j / 2u] * ((j % 2u) == 0u ? 1.0f : -1.0f);
    const float angle = 2.0f * M_PI_F * unit_float(pcg_hash(i ^ 0x6a09e667u));
    const Tangents about = tangents(n);
    const float sine = metal::sqrt(1.0f - cosine * cosine);
    const float3 direction = metal::normalize(cosine * n + sine * (metal::cos(angle) * about.t +
                                                                   metal::sin(angle) * about.b));
    const Hit hit = trace(structure, shapes, leave(surface, direction), serenity::contracts::RayMask::everything);
    result.outward = leaves_outward(surface, direction) ? 1u : 0u;
    result.hit = hit.found ? 1u : 0u;
    result.hit_shape = hit.primitive;
    result.t = hit.t;
    result.cosine = metal::dot(direction, n);
    out[i] = result;
}
