// The packed forms ReSTIR PT stores, on the GPU, each against what it must
// be (kernels/encoding_probe.metal):
//
//   - the equal-area octahedral map (metal/math/warp.metal.h), which packs
//     every unit vector the G-buffer, the light points and the packed BSDF
//     hold: its round trip within the contract's stated angle over the
//     whole sphere, and equal area, uniform draws over the sphere landing
//     uniformly on the square;
//   - the packed BSDF (metal/materials/bsdf.metal.h): every kind back
//     within contract 2's stated tolerances, and the fields a kind does not
//     read 0;
//   - the light point (metal/lights/emitter.metal.h): a light sample's point
//     kept as its identity and evaluated from where it was drawn is the
//     draw's direction, distance, radiance and density.
//
// Every statistical check is deterministic: its numbers are the sampler's,
// keyed by the stream's index, so it passes or fails the same every run.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

#include "core/contracts/bsdf.h"
#include "core/contracts/light_sample.h"
#include "gpu/kernels/encoding_probe.h"
#include "gpu/support/probe_light.h"
#include "gpu/support/probe_runner.h"
#include "support/statistics.h"
#include "support/vector.h"

namespace contracts = serenity::contracts;
namespace tests = serenity::tests;
using tests::Binding;
using tests::chi_square;
using tests::ChiSquare;
using tests::ProbeLight;
using tests::run_with_light;
using tests::significance;
using tests::Vec3;

namespace {

double angle_between(Vec3 a, Vec3 b) {
  return std::acos(
      std::clamp(dot(tests::normalized(a), tests::normalized(b)), -1.0, 1.0));
}

// The angle between `a` and `b`, robust for tiny angles: atan2 of the cross
// product's length and the dot product.
double small_angle(Vec3 a, Vec3 b) {
  return std::atan2(tests::length(cross(a, b)), dot(a, b));
}

const ProbeLight near_light{{0.0f, 2.0f, 0.0f}, 0.5f, 7.0f};

}  // namespace

// ---- the equal-area octahedral map -----------------------------------------

TEST_CASE(
    "the equal-area octahedral map round-trips every direction within the "
    "contract's angle") {
  // A Fibonacci spiral of directions over the whole sphere, and the
  // places the map treats apart: the poles, the equator, the octants'
  // seams, and directions a hair off each.
  std::vector<contracts::Float3> directions;
  constexpr int spiral = 1 << 18;
  const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
  for (int i = 0; i < spiral; ++i) {
    const double z = 1.0 - 2.0 * (i + 0.5) / spiral;
    const double r = std::sqrt(1.0 - z * z);
    directions.push_back(
        tests::float3({r * std::cos(golden * i), r * std::sin(golden * i), z}));
  }
  for (const double hair : {0.0, 1e-7, 1e-5, 1e-3}) {
    for (const int sx : {-1, 1}) {
      for (const int sy : {-1, 1}) {
        for (const int sz : {-1, 1}) {
          for (const Vec3 v : {Vec3{hair, hair, 1.0}, Vec3{1.0, hair, hair},
                               Vec3{hair, 1.0, hair}, Vec3{1.0, 1.0, hair},
                               Vec3{1.0, hair, 1.0}, Vec3{1.0, 1.0, 1.0}}) {
            const Vec3 u =
                tests::normalized(Vec3{sx * v.x, sy * v.y, sz * v.z});
            directions.push_back(tests::float3(u));
          }
        }
      }
    }
  }
  const auto count = static_cast<std::uint32_t>(directions.size());
  tests::ProbeRunner gpu;
  const std::vector<tests::UnitTrip> trips = gpu.run<tests::UnitTrip>(
      "unit_trip", count, {Binding::of(directions), Binding::of(count)});
  double worst = 0.0;
  double worst_length = 0.0;
  for (std::size_t i = 0; i < trips.size(); ++i) {
    worst = std::max(worst, small_angle(tests::vec(directions[i]),
                                        tests::vec(trips[i].decoded)));
    worst_length =
        std::max(worst_length,
                 std::abs(tests::length(tests::vec(trips[i].decoded)) - 1.0));
  }
  MESSAGE("worst round trip " << worst << " radians over " << count
                              << " directions; the contract's "
                              << contracts::unit_round_trip_error);
  CHECK(worst <= contracts::unit_round_trip_error);
  // The bound is the map's, not slack: a map twice as coarse, or a
  // decoding half a cell off, would pass it by no more than this.
  CHECK(worst >= 0.25 * contracts::unit_round_trip_error);
  CHECK(worst_length < 1e-6);
}

TEST_CASE(
    "the equal-area octahedral map is equal-area: uniform draws fill the "
    "square's cells evenly") {
  // 2^22 draws uniform over the sphere, their codes binned 16 x 16 by
  // each coordinate's top four bits: every bin is the same area of the
  // sphere, so each expects 1 / 256 of the draws (the edge bins half a
  // code short, 1e-4 of a bin, far below the noise). And the test can
  // fail: draws by the cosine about +z, which are not uniform, are
  // refused at any significance.
  constexpr std::uint32_t draws = 1u << 22;
  constexpr std::size_t side = 16;
  tests::ProbeRunner gpu;
  const auto binned = [&](std::uint32_t distribution) {
    const std::vector<std::uint32_t> codes = gpu.run<std::uint32_t>(
        "unit_codes", draws,
        {Binding::of(tests::UnitDraws{draws, distribution})});
    std::vector<double> counted(side * side, 0.0);
    for (const std::uint32_t code : codes) {
      counted[((code >> 28u) & 0xfu) * side + ((code >> 12u) & 0xfu)] += 1.0;
    }
    return chi_square(counted, std::vector<double>(
                                   side * side, double{draws} / (side * side)));
  };
  const ChiSquare uniform = binned(tests::draw_uniform);
  MESSAGE("uniform: chi^2 " << uniform.statistic << ", " << uniform.dof
                            << " dof, p " << uniform.p);
  CHECK(uniform.p > significance);
  const ChiSquare cosine = binned(tests::draw_cosine);
  MESSAGE("cosine: chi^2 " << cosine.statistic << ", " << cosine.dof
                           << " dof, p " << cosine.p);
  CHECK(cosine.p < 1e-12);
}

// ---- the packed BSDF -------------------------------------------------------

namespace {

// Contract 2's tolerance for a half: 2^-11 of the value, relatively, from
// half's least normal up; 2^-24, absolutely, below it.
bool within_half(double packed, double value) {
  const double tolerance =
      std::abs(value) >= double{contracts::packed_least_normal}
          ? double{contracts::packed_relative} * std::abs(value)
          : double{contracts::packed_absolute};
  return std::abs(packed - value) <= tolerance;
}

}  // namespace

TEST_CASE(
    "a packed Bsdf unpacks within contract 2's tolerances, for every kind") {
  std::vector<contracts::Bsdf> bsdfs;
  // Normals over the sphere, colors and scalars over the ranges materials
  // resolve to, for each kind.
  for (std::uint32_t i = 0; i < 4096; ++i) {
    const double z = 1.0 - 2.0 * (i + 0.5) / 4096.0;
    const double phi = 2.399963 * i;
    const double r = std::sqrt(1.0 - z * z);
    contracts::Bsdf b{};
    b.normal = tests::float3({r * std::cos(phi), r * std::sin(phi), z});
    const float t = static_cast<float>(i % 97) / 96.0f;
    b.color = {0.02f + 0.97f * t, 0.5f * t + 0.01f, 1.0f - 0.9f * t};
    switch (i % 4) {
      case 0:
        b.kind = contracts::BsdfKind::lambert;
        break;
      case 1:
        b.kind = contracts::BsdfKind::conductor;
        b.alpha = 0.0001f + 0.9f * t * t;
        break;
      case 2:
        b.kind = contracts::BsdfKind::dielectric;
        b.color = {1.0f, 1.0f, 1.0f};
        b.ior = 1.3f + 0.5f * t;
        break;
      default:
        b.kind = contracts::BsdfKind::coated;
        b.ior = 1.4f + 0.3f * t;
        b.escape = 0.9f + 0.09f * t;
        break;
    }
    bsdfs.push_back(b);
  }
  // The contract's endpoints (contracts/bsdf.h), each held to its
  // tolerance in the loop below with the rest: the largest, 65504, in the
  // ior and a color, and its negative; half's least normal, 2^-14, where
  // the relative tolerance starts; and below it, where the absolute one
  // holds: values that round to subnormals, 1.5 x 2^-24, and positive
  // ones under 2^-25, which round to 0 and are raised to 2^-24 (1e-9, 60
  // times smaller, among them).
  const std::size_t endpoints = bsdfs.size();
  contracts::Bsdf largest{};
  largest.normal = {0.0f, 0.0f, 1.0f};
  largest.kind = contracts::BsdfKind::dielectric;
  largest.color = {contracts::packed_largest, -contracts::packed_largest,
                   contracts::packed_least_normal};
  largest.ior = contracts::packed_largest;
  bsdfs.push_back(largest);
  contracts::Bsdf glassy_metal{};
  glassy_metal.normal = {0.0f, 1.0f, 0.0f};
  glassy_metal.kind = contracts::BsdfKind::conductor;
  glassy_metal.color = {1e-9f, 1.5f * contracts::packed_absolute,
                        0.4f * contracts::packed_absolute};
  glassy_metal.alpha = 1e-9f;
  bsdfs.push_back(glassy_metal);
  const auto count = static_cast<std::uint32_t>(bsdfs.size());

  tests::ProbeRunner gpu;
  const std::vector<contracts::Bsdf> back = gpu.run<contracts::Bsdf>(
      "bsdf_round_trip", count, {Binding::of(bsdfs), Binding::of(count)});
  double worst_normal = 0.0;
  std::size_t outside = 0;
  for (std::size_t i = 0; i < bsdfs.size(); ++i) {
    const contracts::Bsdf& a = bsdfs[i];
    const contracts::Bsdf& b = back[i];
    CHECK(a.kind == b.kind);
    worst_normal = std::max(
        worst_normal, small_angle(tests::vec(a.normal), tests::vec(b.normal)));
    const bool colors = within_half(b.color.x, a.color.x) &&
                        within_half(b.color.y, a.color.y) &&
                        within_half(b.color.z, a.color.z);
    const bool scalars = within_half(b.alpha, a.alpha) &&
                         within_half(b.ior, a.ior) &&
                         within_half(b.escape, a.escape);
    outside += colors && scalars ? 0u : 1u;
    // What a kind does not read is 0, padding included.
    CHECK(b.padding[0] == 0u);
    CHECK(b.padding[1] == 0u);
  }
  MESSAGE("worst normal " << worst_normal << " radians");
  CHECK(worst_normal <= contracts::unit_round_trip_error);
  CHECK(outside == 0u);
  // The tolerance check can fail, in either regime: a value off by two
  // halves' steps, and one off by twice the absolute bound.
  CHECK_FALSE(within_half(
      0.75 * (1.0 + 4.0 * double{contracts::packed_relative}), 0.75));
  CHECK_FALSE(within_half(3.0 * double{contracts::packed_absolute}, 1e-9));
  // The endpoints exactly: 65504 and 2^-14 are halves; below 2^-25, 2^-24,
  // never 0.
  const contracts::Bsdf& at_largest = back[endpoints];
  CHECK(at_largest.ior == contracts::packed_largest);
  CHECK(at_largest.color.x == contracts::packed_largest);
  CHECK(at_largest.color.y == -contracts::packed_largest);
  CHECK(at_largest.color.z == contracts::packed_least_normal);
  CHECK(back.back().alpha == contracts::packed_absolute);
  CHECK(back.back().color.x == contracts::packed_absolute);
  CHECK(back.back().color.z == contracts::packed_absolute);
}

// ---- the light point -------------------------------------------------------

TEST_CASE(
    "a light sample's point kept as its identity and evaluated there is the "
    "draw: direction, distance, radiance, density") {
  std::vector<tests::LightPointQuery> queries;
  constexpr int grid = 16;
  for (const contracts::Float3 point : {contracts::Float3{0.0f, 0.0f, 0.0f},
                                        contracts::Float3{1.5f, 0.3f, -0.7f},
                                        contracts::Float3{0.0f, 2.0f, 0.6f}}) {
    for (int i = 0; i < grid * grid; ++i) {
      queries.push_back({point,
                         (static_cast<float>(i % grid) + 0.5f) / grid,
                         (static_cast<float>(i / grid) + 0.5f) / grid,
                         {0u, 0u, 0u}});
    }
  }
  // One inside the light: no sample, as sample_light() draws none.
  queries.push_back({{0.0f, 2.1f, 0.0f}, 0.5f, 0.5f, {0u, 0u, 0u}});
  const auto count = static_cast<std::uint32_t>(queries.size());
  tests::ProbeRunner gpu;
  const std::vector<tests::LightPointResult> results =
      run_with_light<tests::LightPointResult>(
          gpu, "light_points", count, near_light,
          {Binding::of(queries), Binding::of(count)});

  const double radius = near_light.radius;
  for (std::size_t i = 0; i + 1 < results.size(); ++i) {
    const tests::LightPointResult& r = results[i];
    INFO("query " << i);
    REQUIRE(contracts::holds_light(contracts::LightPoint{r.light, 0u}));
    CHECK(contracts::light_index(contracts::LightPoint{r.light, 0u}) == 0u);
    // The point is its cell's: within the round trip's angle on the unit
    // sphere, r times it in the world, seen from the point at its
    // distance.
    const double moved = radius * contracts::unit_round_trip_error;
    CHECK(angle_between(tests::vec(r.direction),
                        tests::vec(r.sample_direction)) <=
          1.01 * moved / r.sample_distance + 1e-6);
    CHECK(std::abs(r.distance - r.sample_distance) <= 1.01 * moved + 1e-6);
    CHECK(r.radiance.x == r.sample_radiance.x);
    CHECK(r.radiance.y == r.sample_radiance.y);
    CHECK(r.pdf == doctest::Approx(r.sample_pdf).epsilon(1e-6));
    CHECK(r.area_scale == doctest::Approx(radius * radius).epsilon(1e-6));
    CHECK(r.cos_light > 0.0f);
  }
  CHECK_FALSE(
      contracts::holds_light(contracts::LightPoint{results.back().light, 0u}));
  CHECK(results.back().sample_pdf == 0.0f);
}
