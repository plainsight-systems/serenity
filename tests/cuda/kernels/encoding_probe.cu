// Runs the packed forms ReSTIR PT stores on their own for
// tests/gpu/encoding_test.cpp: the equal-area octahedral map
// (cuda/math/warp.cuh), the packed BSDF (metal/materials/
// metal/kernels/bsdf.metal.h) and the light point (cuda/lights/emitter.cuh).
// Bindings: the output at 0, the inputs after it in the order each kernel
// lists them (encoding_probe.h). The light kernel's light is one sphere
// light, record 0, read through the emitter as the passes read them: the
// records, the spheres, the transforms, the glows and which light each
// shape is, at 1 to 5 (gpu/support/probe_light.h).

#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "core/lights/light.h"
#include "core/lights/sphere_light.h"
#include "cuda/lights/emitter.cuh"
#include "cuda/materials/bsdf.cuh"
#include "cuda/math/vector.cuh"
#include "cuda/math/warp.cuh"
#include "cuda/sampler/sampler.cuh"
#include "gpu/kernels/encoding_probe.h"

namespace serenity {
namespace shaders {

using serenity::contracts::Bsdf;
using serenity::contracts::LightPoint;
using serenity::tests::LightPointQuery;
using serenity::tests::LightPointResult;
using serenity::tests::UnitDraws;
using serenity::tests::UnitTrip;

// Each direction's code, and the code decoded. Inputs: the directions, their
// count.
extern "C" __global__ void unit_trip(
    UnitTrip* out, const serenity::contracts::Float3* directions,
    const uint* count_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  if (i >= count) {
    return;
  }
  const uint code = encode_unit(to_float3(directions[i]));
  out[i] = UnitTrip{to_packed(decode_unit(code)), code};
}

// Draw i of a distribution over the sphere, from numbers hashed from i, as
// its code. Input: the draws' query.
extern "C" __global__ void unit_codes(uint* out, const UnitDraws* draws_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const UnitDraws& draws = *draws_at;
  if (i >= draws.count) {
    return;
  }
  PathNumbers numbers =
      path_numbers(make_uint2(i & 0xffffu, i >> 16u), 0x5eed5u);
  const float2 u = next_numbers2(numbers);
  const float3 d =
      draws.distribution == serenity::tests::draw_cosine
          ? cosine_direction(make_float3(0.0f, 0.0f, 1.0f), u)
          : make_float3(sqrt(max(0.0f, 1.0f - (1.0f - 2.0f * u.x) *
                                                  (1.0f - 2.0f * u.x))) *
                            cos(2.0f * pi * u.y),
                        sqrt(max(0.0f, 1.0f - (1.0f - 2.0f * u.x) *
                                                  (1.0f - 2.0f * u.x))) *
                            sin(2.0f * pi * u.y),
                        1.0f - 2.0f * u.x);
  out[i] = encode_unit(d);
}

// Each Bsdf packed and unpacked. Inputs: the Bsdfs, their count.
extern "C" __global__ void bsdf_round_trip(Bsdf* out, const Bsdf* bsdfs,
                                           const uint* count_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  if (i < count) {
    out[i] = unpack_bsdf(pack_bsdf(bsdfs[i]));
  }
}

// From each query's point with its u: sample_light(); then the point that
// direction reaches on the light kept as its identity, light_point_at(), and
// evaluated from the same point, evaluate_light_point() and
// light_point_pdf(), as ReSTIR PT keeps a NEE candidate's point
// (metal/integrator/nee.metal.h, kept_point). Inputs: the light (1 to 5),
// the queries, their count.
extern "C" __global__ void light_points(
    LightPointResult* out, const serenity::lights::LightRecord* records,
    const serenity::lights::SphereLightData* spheres,
    const serenity::contracts::Transform* transforms, const float* glows,
    const uint* shape_lights, const LightPointQuery* queries,
    const uint* count_at) {
  const uint i = blockIdx.x * blockDim.x + threadIdx.x;
  const uint& count = *count_at;
  if (i >= count) {
    return;
  }
  const Lights lights{records, shape_lights, spheres, transforms, glows};
  const float3 point = to_float3(queries[i].point);
  const float2 u = make_float2(queries[i].u0, queries[i].u1);
  const serenity::contracts::LightSample drawn =
      sample_light(lights, records[0], point, u);
  LightPointResult result{};
  result.sample_direction = drawn.direction;
  result.sample_distance = drawn.distance;
  result.sample_radiance = drawn.radiance;
  result.sample_pdf = drawn.pdf;
  if (drawn.pdf > 0.0f) {
    const PlacedLight light = place_light(lights, 0u);
    const float3 reached = point + drawn.distance * to_float3(drawn.direction);
    const LightPoint y = light_point_at(light, 0u, reached);
    const serenity::contracts::LightReceived received =
        evaluate_light_point(light, y, point);
    result.light = y.light;
    result.direction = received.direction;
    result.distance = received.distance;
    result.radiance = received.radiance;
    result.cos_light = received.cos_light;
    result.area_scale = received.area_scale;
    result.pdf = light_point_pdf(light, point, received);
  }
  out[i] = result;
}

}  // namespace shaders
}  // namespace serenity
