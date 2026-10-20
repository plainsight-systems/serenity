// Runs the sphere light's shader half (cuda/lights/sphere_light.cuh)
// for tests/gpu/light_test.cpp. The light is its data and its shape's
// transform, as the emitter reads it (metal/kernels/emitter.metal.h). Bindings:
// the output at 0, then the light's data, the query and the light's transform
// (light_probe.h).

#include "core/contracts/emitter.h"
#include "core/lights/sphere_light.h"
#include "cuda/device/layout.cuh"
#include "cuda/lights/sphere_light.cuh"
#include "cuda/math/hash.cuh"
#include "cuda/math/vector.cuh"
#include "gpu/kernels/light_probe.h"

namespace serenity {
namespace shaders {

using serenity::tests::LightDraw;
using serenity::tests::LightFar;
using serenity::tests::LightQuery;
using serenity::tests::LightSampleDraw;

// The direction drawn from sample i of a query.grid x query.grid grid over
// the unit square, and the distance along it to the light's surface.
extern "C" __global__ void light_cone(
    LightDraw* out, const serenity::lights::SphereLightData* data_at,
    const LightQuery* query_at,
    const serenity::contracts::Transform* placed_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const serenity::lights::SphereLightData& data = *data_at;
  const LightQuery& query = *query_at;
  const serenity::contracts::Transform& placed = *placed_at;
  const SphereLight light = sphere_light(data, placed, 1.0f);
  const uint n = query.grid;
  if (i >= n * n) {
    return;
  }
  const LightView view = view_light(light, to_float3(query.point));
  const float2 u = (make_float2(i % n, i / n) + 0.5f) / float(n);
  const float3 d = direction_to_light(view, u);
  out[i] = LightDraw{to_packed(d), distance_to_light(light, view, d)};
}

// For a light seen from far off: the emitter's sample's pdf, and
// light_pdf() asked of the direction to the light's middle and of one well
// outside it.
extern "C" __global__ void light_far(
    LightFar* out, const serenity::lights::SphereLightData* data_at,
    const LightQuery* query_at,
    const serenity::contracts::Transform* placed_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const serenity::lights::SphereLightData& data = *data_at;
  const LightQuery& query = *query_at;
  const serenity::contracts::Transform& placed = *placed_at;
  const SphereLight light = sphere_light(data, placed, 1.0f);
  if (i != 0) {
    return;
  }
  const float3 point = to_float3(query.point);
  const serenity::contracts::LightSample s =
      sphere_sample_light(light, point, make_float2(0.25f, 0.5f));
  const float3 middle = normalize(light.center - point);
  const float3 aside = normalize(middle + make_float3(0.01f, 0.0f, 0.0f));
  out[0] = LightFar{s.pdf, sphere_light_pdf(light, point, middle),
                    sphere_light_pdf(light, point, aside), s.distance};
}

// Draw i of query.grid draws of the emitter's sample() from query.point,
// from numbers hashed from i, as a path draws them; and pdf() asked of the
// drawn direction.
extern "C" __global__ void light_samples(
    LightSampleDraw* out, const serenity::lights::SphereLightData* data_at,
    const LightQuery* query_at,
    const serenity::contracts::Transform* placed_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const serenity::lights::SphereLightData& data = *data_at;
  const LightQuery& query = *query_at;
  const serenity::contracts::Transform& placed = *placed_at;
  if (i >= query.grid) {
    return;
  }
  const SphereLight light = sphere_light(data, placed, 1.0f);
  const float3 point = to_float3(query.point);
  const uint h = pcg_hash(i ^ 0x5bd1e995u);
  const serenity::contracts::LightSample s = sphere_sample_light(
      light, point, make_float2(unit_float(h), unit_float(pcg_hash(h))));
  out[i] =
      LightSampleDraw{s.direction,
                      s.pdf,
                      s.radiance,
                      s.distance,
                      sphere_light_pdf(light, point, to_float3(s.direction)),
                      {0u, 0u, 0u}};
}

}  // namespace shaders
}  // namespace serenity
