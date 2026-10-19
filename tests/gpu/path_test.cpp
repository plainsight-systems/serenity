// The path tracer (metal/integrator/path.metal.h, metal/passes/path/), on
// the GPU: the renderer of the training targets and of every reference, so
// its mean is what everything else is held to.
//
//   - an integration test, converged over frames and checked against what
//     it must converge to, worked out from first principles: under a sky of
//     one radiance L, a white Lambert sphere, a glass sphere and a white
//     coated one each return every bit of light they receive, so every
//     pixel converges to L (a wrong weight, or a doubled count, shows); and
//     a floor under one firefly in a black sky converges to albedo L sin^2
//     cos, the uniform selection's 1 / N, the emitter's pdf and principle 7
//     (the light sample and the bounce that reaches a firefly, each
//     weighted, counting it once between them) all at once. Every converged
//     render also checks that no sample was left out for not being finite
//     (metal/film/failure_count.h);
//   - unit tests of multiple importance sampling's weights (principle 7), on
//     the probe kernel's draws of both strategies from the path tracer's own
//     functions: they sum to 1 for a direction both draw, from the points
//     each asks its density from, at a firefly far off and one millimeters
//     across and near; and the power heuristic is finite and complementary
//     at any densities a float holds.

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "core/contracts/bsdf.h"
#include "core/scene/scene.h"
#include "gpu/kernels/path_probe.h"
#include "gpu/support/probe_light.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "support/lit_floor.h"
#include "support/scene_text.h"
#include "support/vector.h"

namespace contracts = serenity::contracts;
namespace frame = serenity::frame;
namespace tests = serenity::tests;
using tests::expected_floor;
using tests::furnace;
using tests::lit_floor;
using tests::middle_of;
using tests::one_firefly;
using tests::Patch;
using tests::patch_mean;

namespace {

// The path graph's accumulating rig (gpu/support/rendering.h).
struct Rig final : tests::AccumulatingRig {
  Rig(std::string_view text, frame::Extent size)
      : AccumulatingRig(text, tests::path_graph(), size) {}
};

constexpr frame::Extent square{64, 64};
constexpr Patch middle = middle_of(square, 6);

}  // namespace

TEST_CASE(
    "the path tracer converges to closed forms: three furnaces, and a floor "
    "under a firefly") {
  SUBCASE("a white Lambert sphere under a uniform sky converges to the sky") {
    // Albedo 1 and a sky of 0.5 every way: what the sphere receives it
    // returns. The sphere's middle, and the sky beside it.
    Rig rig(furnace("kind = \"rough\"\ncolor = [1, 1, 1]\n"), square);
    const auto image = rig.render_accumulated(1024);
    CHECK(patch_mean(image, rig.size(), middle) ==
          doctest::Approx(0.5).scale(0).epsilon(0.02));
    CHECK(patch_mean(image, rig.size(), {3, 3, 2}) ==
          doctest::Approx(0.5).scale(0).epsilon(0.01));
  }
  SUBCASE(
      "a glass sphere: reflected or refracted, every path leaves with the "
      "sky's radiance") {
    Rig rig(furnace("kind = \"dielectric\"\nior = 1.5\n"), square);
    const auto image = rig.render_accumulated(512);
    CHECK(patch_mean(image, rig.size(), middle) ==
          doctest::Approx(0.5).scale(0).epsilon(0.02));
  }
  SUBCASE(
      "a white coated sphere: the coat's share and the base's "
      "(materials/coated.h)") {
    Rig rig(furnace("kind = \"coated\"\ncolor = [1, 1, 1]\nior = 1.5\n"),
            square);
    const auto image = rig.render_accumulated(1024);
    CHECK(patch_mean(image, rig.size(), middle) ==
          doctest::Approx(0.5).scale(0).epsilon(0.02));
  }
  SUBCASE("a floor under one firefly converges to albedo L sin^2 cos") {
    Rig rig(lit_floor(one_firefly), {64, 48});
    const auto image = rig.render_accumulated(512);
    for (const tests::Pixel pixel :
         {tests::Pixel{32, 30}, tests::Pixel{14, 38}, tests::Pixel{50, 26}}) {
      // A 3 x 3 patch's mean against the formula at its center: the
      // formula varies little over three pixels of floor.
      const double expected = expected_floor(rig.description(), rig.size(),
                                             pixel.x + 0.5, pixel.y + 0.5);
      INFO("pixel " << pixel.x << ", " << pixel.y << ": expected " << expected);
      CHECK(patch_mean(image, rig.size(), {pixel.x, pixel.y, 1}) ==
            doctest::Approx(expected).scale(0).epsilon(0.03));
    }
  }
}

// ---- multiple importance sampling (path.metal.h, steps 3 and 5) -----------

namespace {

// A surface at the origin facing +y, and fireflies where a mirror there
// would show them to wo: what the MIS probe asks about. mis_light is 0.3
// across at 1.6, near_light 3 mm across at 2 cm, near enough that the step
// a ray leaves the surface from (integrator/surface.metal.h, 15 um at the
// origin) moves the cone the light fills.
const contracts::Float3 mis_wo{0.0f, 0.5547002f,
                               0.8320503f};  // (0, 2, 3) normalized
const tests::ProbeLight mis_light{{0.0f, 0.9f, -1.35f}, 0.3f, 10.0f};
const tests::ProbeLight near_light{{0.0f, 0.011094004f, -0.016641006f},
                                   0.003f,
                                   10.0f};  // 0.02 along (0, 2, -3)

contracts::Bsdf floor_bsdf(contracts::BsdfKind kind, float alpha) {
  contracts::Bsdf bsdf{};
  bsdf.normal = {0.0f, 1.0f, 0.0f};
  bsdf.kind = kind;
  bsdf.color = {0.8f, 0.8f, 0.8f};
  bsdf.alpha = alpha;
  return bsdf;
}

// What the MIS probe found over its draws of both strategies.
struct WeightSums {
  std::size_t both = 0;  // directions both strategies draw
  std::size_t by_bsdf_both =
      0;  // of them, the BSDF's draws: bounces that strike the light
  double worst = 0.0;  // the largest |w_l + w_b - 1| among them
  double worst_alone =
      0.0;  // the largest |w_b - 1| where only the BSDF draws the direction

  double worst_either() const { return std::max(worst, worst_alone); }
};

// Where step 5 draws from, for the probe: as the path tracer draws, or from
// the surface's point, as 7751a65 did.
enum class LightFrom : std::uint8_t { path_tracer = 0, surface_point = 1 };

// How the probe's two steps weigh: step 5 choosing among `light_count`
// lights, step 3 weighing the light a bounce strikes as one of
// `struck_light_count`, step 5 drawing `from` (I.24: two counts by name).
struct Weighing {
  std::uint32_t light_count = 0;
  std::uint32_t struck_light_count = 0;
  LightFrom from = LightFrom::path_tracer;
};

WeightSums weight_sums(tests::ProbeRunner& gpu, const contracts::Bsdf& bsdf,
                       const tests::ProbeLight& light, Weighing weighing) {
  constexpr std::uint32_t draws = 4096;
  const tests::MisQuery query{{0.0f, 0.0f, 0.0f},
                              draws,
                              mis_wo,
                              weighing.light_count,
                              {0.0f, 1.0f, 0.0f},
                              weighing.struck_light_count,
                              static_cast<std::uint32_t>(weighing.from),
                              {0u, 0u, 0u}};
  const std::vector<tests::MisDraw> results =
      tests::run_with_light<tests::MisDraw>(
          gpu, "mis_weights", 2 * draws, light,
          {tests::Binding::of(query), tests::Binding::of(bsdf)});
  WeightSums sums;
  for (const tests::MisDraw& d : results) {
    if (d.light_density > 0.0f && d.bsdf_density > 0.0f) {
      ++sums.both;
      sums.by_bsdf_both += d.by_bsdf;
      sums.worst = std::max(
          sums.worst,
          std::abs(static_cast<double>(d.light_weight) + d.bsdf_weight - 1.0));
    } else if (d.by_bsdf == 1u && d.bsdf_density > 0.0f) {
      sums.worst_alone = std::max(
          sums.worst_alone, std::abs(static_cast<double>(d.bsdf_weight) - 1.0));
    }
  }
  return sums;
}

}  // namespace

TEST_CASE(
    "MIS: for a direction both strategies draw, the light sample's weight and "
    "the bounce's sum to 1") {
  // From the densities each step asks, from the point each asks them from
  // (path.metal.h, "One origin"): step 5 the selection's 1 / N, the
  // light's own sample pdf from next_event_origin() and the BSDF's pdf();
  // step 3 the bounce's density as step 6 hands it on (handed_on()), the
  // selection's 1 / N and light_pdf() from the point the bounce leaves
  // from (leave()). At a glossy conductor and a Lambert surface, under a
  // firefly 0.3 across at 1.6 and one 3 mm across at 2 cm, over each
  // strategy's draws. The sum is 1 only where the two steps ask the same
  // densities of the same direction from the same point; to 10^-4, the
  // rounding of a conductor's half vector, found from the drawn
  // microfacet normal by sample() and from wo + wi by pdf(). A direction
  // only the BSDF draws, one that misses the light, is the bounce's
  // alone: w_b 1, to the rounding of fast math's division.
  constexpr double tolerance = 1e-4;
  tests::ProbeRunner gpu;
  const std::array<std::pair<std::string, contracts::Bsdf>, 2> surfaces{
      {{"conductor, alpha 0.1",
        floor_bsdf(contracts::BsdfKind::conductor, 0.1f)},
       {"lambert", floor_bsdf(contracts::BsdfKind::lambert, 0.0f)}}};
  const std::array<std::pair<std::string, tests::ProbeLight>, 2> lights{
      {{"firefly 0.3 across at 1.6", mis_light},
       {"firefly 3 mm across at 2 cm", near_light}}};
  for (const auto& [light_name, light] : lights) {
    for (const auto& [name, bsdf] : surfaces) {
      const WeightSums sums = weight_sums(
          gpu, bsdf, light, {.light_count = 4, .struck_light_count = 4});
      INFO(light_name << ", " << name << ": " << sums.both
                      << " directions both draw (" << sums.by_bsdf_both
                      << " of them the BSDF's), worst |w_l + w_b - 1| "
                      << sums.worst << ", worst |w_b - 1| alone "
                      << sums.worst_alone);
      MESSAGE(light_name << ", " << name << ": " << sums.both
                         << " directions both draw (" << sums.by_bsdf_both
                         << " of them the BSDF's), worst |w_l + w_b - 1| "
                         << sums.worst << ", worst |w_b - 1| alone "
                         << sums.worst_alone);
      // Not empty: every light sample is above the surface, and some
      // bounces strike the light.
      CHECK(sums.both > 4096);
      CHECK(sums.by_bsdf_both > 0);
      CHECK(sums.worst < tolerance);
      CHECK(sums.worst_alone < 1e-6);
      // And the check can fail: step 3 weighing the struck light as
      // if it were the only one (P = 1) while step 5 chose among 4.
      // Off by far more than the tolerance at either surface: by most
      // of the weight at the conductor, where the strategies share
      // it; less at the Lambert surface, where the light's density
      // dominates both.
      const WeightSums wrong = weight_sums(
          gpu, bsdf, light, {.light_count = 4, .struck_light_count = 1});
      MESSAGE(
          light_name
          << ", " << name
          << ", step 3's P of 1 against step 5's 1 / 4: worst |w_l + w_b - 1| "
          << wrong.worst);
      CHECK(wrong.worst > 10.0 * tolerance);
    }
  }
  // And it fails where step 5 draws from the surface's point, as 7751a65
  // drew, while step 3 asks from the point the bounce leaves from: at the
  // near firefly, whose cone the 15 um between the two points moves, the
  // two densities of one direction differ, and in the crescent one
  // point's cone holds and the other's does not, only one is 0. At the far
  // firefly the two points' densities differ within the rounding: why the
  // check above asks at the near one too.
  for (const auto& [name, bsdf] : surfaces) {
    const WeightSums wrong = weight_sums(gpu, bsdf, near_light,
                                         {.light_count = 4,
                                          .struck_light_count = 4,
                                          .from = LightFrom::surface_point});
    MESSAGE("firefly 3 mm across at 2 cm, "
            << name
            << ", step 5 from the surface's point: worst |w_l + w_b - 1| "
            << wrong.worst << ", worst |w_b - 1| alone " << wrong.worst_alone);
    CHECK(wrong.worst_either() > 10.0 * tolerance);
  }
}

TEST_CASE(
    "MIS: the power heuristic is finite and complementary at any densities a "
    "float holds") {
  // Densities from 0 to infinity, the pairs the path tracer can meet: a
  // light's P p_l past 1.8 x 10^19 (a sphere of 10^-10 m seen from a
  // meter: 3.2 x 10^19), whose square a float does not hold; the largest
  // float; an infinite density beside a finite one, and two infinite; a
  // density beside 0; and ordinary ones. Each weight is checked against
  // f^2 / (f^2 + g^2) worked out in doubles (its limit where a density is
  // infinite, 1/2 for two), and the two ways round must sum to 1.
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr float largest = std::numeric_limits<float>::max();
  const std::vector<tests::HeuristicQuery> queries{
      {2.0f, 1.0f},       {1.0f, 1.0f},   {0.3f, 0.0f},   {3.2e19f, 0.3f},
      {0.3f, 3.2e19f},    {1e20f, 1e20f}, {1e30f, 3e29f}, {largest, 1e-3f},
      {largest, largest}, {inf, 1.0f},    {1.0f, inf},    {inf, inf},
      {1e-30f, 1e-20f},   {1e-20f, 0.0f}};
  tests::ProbeRunner gpu;
  const auto count = static_cast<std::uint32_t>(queries.size());
  const std::vector<tests::HeuristicWeights> weights =
      gpu.run<tests::HeuristicWeights>(
          "heuristic_weights", count,
          {tests::Binding::of(queries), tests::Binding::of(count)});
  std::size_t squared_not_finite = 0;
  for (std::size_t k = 0; k < queries.size(); ++k) {
    const double f = queries[k].drawn;
    const double g = queries[k].other;
    double expected = 0.0;
    if (std::isinf(f) && std::isinf(g)) {
      expected = 0.5;
    } else if (std::isinf(f)) {
      expected = 1.0;
    } else if (std::isinf(g)) {
      expected = 0.0;
    } else {
      expected = f * f / (f * f + g * g);
    }
    const tests::HeuristicWeights& w = weights[k];
    INFO("densities " << f << ", " << g << ": weight " << w.weight
                      << " (expected " << expected << "), swapped " << w.swapped
                      << ", squared directly " << w.squared);
    CHECK(std::isfinite(w.weight));
    CHECK(std::isfinite(w.swapped));
    // Absolute: a weight in [0, 1] to 10^-6 (a float flushes a weight
    // under 10^-38 to 0, as at 0.3 beside 3.2 x 10^19).
    CHECK(w.weight == doctest::Approx(expected).epsilon(1e-6));
    CHECK(static_cast<double>(w.weight) + w.swapped ==
          doctest::Approx(1.0).scale(0).epsilon(1e-6));
    squared_not_finite += std::isfinite(w.squared) ? 0 : 1;
  }
  // And the densities asked are ones that break the heuristic squared
  // directly, as 7751a65 computed it: NaN at some of them.
  MESSAGE(squared_not_finite << " of " << queries.size()
                             << " pairs not finite when squared directly");
  CHECK(squared_not_finite >= 5);
}
