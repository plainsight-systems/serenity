// Runs the tracing loop (metal/acceleration/trace.metal.h) over the scene's
// structures, for tests/gpu/acceleration_test.cpp: thread i traces ray i,
// carrying its mask, and writes the hit's t and its shape, or found 0, and
// whether a shadow ray along it, which carries the occluders, is blocked within
// its reach. What it checks: a world ray carried into a shape's object
// space by its transform keeps the world's t, at any scale, and a hit names
// its shape by its primitive index
// (scene_acceleration.h), and a mask finds its classes' shapes alone
// (contract 19). Bindings: the output at 0, then the rays, their count,
// the structures' block, and the shapes' records, transforms and boxes
// (trace_probe.h).

#include <metal_raytracing>
#include <metal_stdlib>

#include "core/contracts/emitter.h"
#include "core/contracts/transform.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "metal/acceleration/trace.metal.h"
#include "metal/device/layout.metal.h"
#include "trace_probe.h"

using namespace serenity::shaders;
using serenity::tests::TraceHit;
using serenity::tests::TraceRay;

kernel void trace_probe(device TraceHit* out [[buffer(0)]],
                        device const TraceRay* rays [[buffer(1)]],
                        constant uint& count [[buffer(2)]],
                        SceneStructure structure [[buffer(3)]],
                        constant serenity::shapes::ShapeRecord* records
                        [[buffer(4)]],
                        constant serenity::contracts::Transform* transforms
                        [[buffer(5)]],
                        constant serenity::shapes::BoxData* boxes [[buffer(6)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= count) {
    return;
  }
  const Shapes shapes{records, transforms, boxes};
  const TraceRay r = rays[i];
  const Hit hit =
      trace(structure, shapes,
            metal::raytracing::ray(to_float3(r.origin), to_float3(r.direction),
                                   0.0f, unbounded),
            r.mask);
  const bool blocked =
      occluded(structure, shapes,
               metal::raytracing::ray(to_float3(r.origin),
                                      to_float3(r.direction), 0.0f, r.reach));
  out[i] = TraceHit{hit.found ? hit.t : 0.0f, hit.found ? hit.primitive : 0u,
                    hit.found ? 1u : 0u, blocked ? 1u : 0u};
}
