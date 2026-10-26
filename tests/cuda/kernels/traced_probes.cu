// The CUDA backend's tracing probes, one OptiX module (tests/cuda/
// CMakeLists.txt), each a ray generation program run by the tracing
// harness (tests/cuda/support/traced_probe.h) and checked by the test it
// shares with Metal: Metal's trace_probe, leave_probe and seed_probe
// (tests/gpu/kernels/), their bindings read from the launch parameters at
// the places Metal binds them (probe_launch.h), the structures among them.
// The module holds the backend's one intersection program, MNEE's seed's,
// as every tracing module does (cuda/frame/traced_module.cuh).

#include <optix.h>

#include "core/contracts/emitter.h"
#include "core/contracts/transform.h"
#include "core/scene/scene_block.h"
#include "core/shapes/box.h"
#include "core/shapes/primitive.h"
#include "cuda/acceleration/trace.cuh"
#include "cuda/device/layout.cuh"
#include "cuda/integrator/mnee.cuh"
#include "cuda/integrator/mnee_seed.cuh"
#include "cuda/integrator/surface.cuh"
#include "cuda/materials/resolve.cuh"
#include "cuda/math/hash.cuh"
#include "cuda/math/warp.cuh"
#include "cuda/shapes/shapes.cuh"
#include "gpu/kernels/leave_probe.h"
#include "gpu/kernels/seed_probe.h"
#include "gpu/kernels/trace_probe.h"
#include "kernels/probe_launch.h"

extern "C" {
__constant__ serenity::tests::ProbeLaunch probe;
}

namespace serenity {
namespace shaders {
namespace {

// Binding `i`, as Metal's probe declares it.
template <typename T>
__device__ inline T* binding(uint i) {
  return reinterpret_cast<T*>(probe.bindings[i]);
}

// The scene's shapes, at bindings 4 to 6 of both probes, after the
// structures' place at 3.
__device__ inline Shapes probe_shapes() {
  return Shapes{binding<const serenity::shapes::ShapeRecord>(4),
                binding<const serenity::contracts::Transform>(5),
                binding<const serenity::shapes::BoxData>(6)};
}

// Thread i traces ray i, carrying its mask, and writes the hit's t and its
// shape, or found 0, and whether a shadow ray along it carrying the same
// mask is blocked within its reach (Metal's trace_probe, which says what it
// checks). Bindings: the output at 0, then the rays, their count, the
// structures (here in the launch parameters), and the shapes' records,
// transforms and boxes.
__device__ inline void trace_one() {
  using serenity::tests::TraceHit;
  using serenity::tests::TraceRay;
  const uint i = optixGetLaunchIndex().x;
  if (i >= *binding<const uint>(2)) {
    return;
  }
  const Shapes shapes = probe_shapes();
  const TraceRay r = binding<const TraceRay>(1)[i];
  const Hit hit = trace(
      &probe.structures, shapes,
      make_ray(to_float3(r.origin), to_float3(r.direction), 0.0f, unbounded),
      r.mask);
  const bool blocked = occluded(
      &probe.structures, shapes,
      make_ray(to_float3(r.origin), to_float3(r.direction), 0.0f, r.reach));
  binding<TraceHit>(0)[i] =
      TraceHit{hit.found ? hit.t : 0.0f, hit.found ? hit.primitive : 0u,
               hit.found ? 1u : 0u, blocked ? 1u : 0u};
}

// The cosines to the normal the leaving rays take, head on to grazing.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays): a device table
static constexpr __device__ float
    leave_cosines[serenity::tests::leave_cosine_count] = {
        1.0f, 0.5f, 0.1f, 0.03f, 0.01f, 3e-3f, 1e-3f, 3e-4f};

// Metal's leave_probe (tests/gpu/kernels/leave_probe.metal, which says
// what it checks): thread i takes query i / leave_directions, traces its
// ray, finds the surface at its hit, and traces one ray leaving it by
// leave(). Bindings as trace_one()'s.
__device__ inline void leave_one() {
  using serenity::tests::LeaveQuery;
  using serenity::tests::LeaveResult;
  const uint i = optixGetLaunchIndex().x;
  if (i >= *binding<const uint>(2) * serenity::tests::leave_directions) {
    return;
  }
  const Shapes shapes = probe_shapes();
  const LeaveQuery query =
      binding<const LeaveQuery>(1)[i / serenity::tests::leave_directions];
  const uint j = i % serenity::tests::leave_directions;
  const Ray aimed = make_ray(to_float3(query.origin),
                             to_float3(query.direction), 0.0f, unbounded);
  const Hit found = trace(&probe.structures, shapes, aimed,
                          serenity::contracts::RayMask::everything);
  LeaveResult result{};
  LeaveResult* const out = binding<LeaveResult>(0);
  if (!found.found || found.primitive != query.shape) {
    out[i] = result;
    return;
  }
  result.found = 1u;
  result.found_part = optixHitObjectGetPrimitiveIndex();  // its triangle
  // The surface as the path tracer finds it: the hit triangle's point
  // (shapes/shapes.cuh).
  const serenity::contracts::SurfaceInteraction surface =
      surface_interaction(shapes, found.primitive, found.at, aimed.direction);
  const float3 n = to_float3(surface.geometric_normal);
  const float cosine = leave_cosines[j / 2u] * ((j % 2u) == 0u ? 1.0f : -1.0f);
  const float angle = 2.0f * pi * unit_float(pcg_hash(i ^ 0x6a09e667u));
  const Tangents about = tangents(n);
  const float sine = sqrt(1.0f - cosine * cosine);
  const float3 direction = normalize(
      cosine * n + sine * (cos(angle) * about.t + sin(angle) * about.b));
  const Hit hit = trace(&probe.structures, shapes, leave(surface, direction),
                        serenity::contracts::RayMask::everything);
  result.hit_part = hit.found ? optixHitObjectGetPrimitiveIndex() : 0u;
  result.outward = leaves_outward(surface, direction) ? 1u : 0u;
  result.hit = hit.found ? 1u : 0u;
  result.hit_shape = hit.primitive;
  result.t = hit.t;
  result.cosine = dot(direction, n);
  out[i] = result;
}

// What seed_segment() reads of a scene: the structures, the shapes and the
// materials (Metal's seed_probe's SeedScene).
struct SeedScene {
  SceneStructure structure;
  Shapes traced;
  Materials kinds;
  __device__ Shapes shapes() const { return traced; }
  __device__ Materials materials() const { return kinds; }
};

// Thread i seeds segment i with this backend's seed (mnee.cuh,
// seed_segment(), its walk the seed's intersection program, mnee_seed.cuh)
// and writes the seed (Metal's seed_probe, which says what it checks).
// Bindings: the output at 0, then the segments, their count, the scene's
// block, the structures (here in the launch parameters), and the shapes'
// records, transforms and boxes, at 5 to 7.
__device__ inline void seed_one() {
  using serenity::tests::SeedFound;
  using serenity::tests::SeedSegment;
  const uint i = optixGetLaunchIndex().x;
  if (i >= *binding<const uint>(2)) {
    return;
  }
  const serenity::gpu::SceneBlock& block =
      *binding<const serenity::gpu::SceneBlock>(3);
  const SeedScene scene{
      &probe.structures,
      Shapes{binding<const serenity::shapes::ShapeRecord>(5),
             binding<const serenity::contracts::Transform>(6),
             binding<const serenity::shapes::BoxData>(7)},
      Materials{block.materials, block.roughs, block.dielectrics,
                block.conductors, block.coateds}};
  const SeedSegment s = binding<const SeedSegment>(1)[i];
  // No shape to ignore: the seed traces the occluders, which hold no light.
  const mnee::Seed seed =
      mnee::seed_segment(scene, to_float3(s.origin), to_float3(s.direction),
                         s.distance, serenity::contracts::no_primitive);
  SeedFound found{};
  found.count = seed.count;
  found.blocked = seed.blocked ? 1u : 0u;
  found.usable = seed.usable ? 1u : 0u;
  for (uint c = 0u; c < mnee::max_interfaces; ++c) {
    found.t[c] = c < seed.count ? seed.t[c] : 0.0f;
    found.shape[c] = c < seed.count ? seed.shape[c] : 0u;
  }
  binding<SeedFound>(0)[i] = found;
}

}  // namespace
}  // namespace shaders
}  // namespace serenity

extern "C" __global__ void __raygen__seed_probe() {
  serenity::shaders::seed_one();
}

extern "C" __global__ void __raygen__trace_probe() {
  serenity::shaders::trace_one();
}

extern "C" __global__ void __raygen__leave_probe() {
  serenity::shaders::leave_one();
}
