// Runs the shapes' surface interaction (metal/shapes/shapes.metal.h) for
// tests/gpu/shape_test.cpp: thread i fills contract 1 at queries[i].point on
// shape queries[i].shape, a ray arriving from +z. Bindings: the output at 0,
// then the queries, their count, and the shapes' records, transforms and
// boxes (probes.h).

#include <metal_stdlib>

#include "metal/device/layout.metal.h"
#include "metal/shapes/shapes.metal.h"
#include "probes.h"

using namespace serenity::shaders;
using serenity::tests::ShapeQuery;

kernel void interaction_probe(device serenity::contracts::SurfaceInteraction* out [[buffer(0)]],
                              device const ShapeQuery* queries [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              constant serenity::shapes::ShapeRecord* records [[buffer(3)]],
                              constant serenity::contracts::Transform* transforms [[buffer(4)]],
                              constant serenity::shapes::BoxData* boxes [[buffer(5)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < count) {
        out[i] = surface_interaction(Shapes{records, transforms, boxes}, queries[i].shape,
                                     to_float3(queries[i].point), float3(0.0f, 0.0f, -1.0f));
    }
}
