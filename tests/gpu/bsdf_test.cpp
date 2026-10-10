// Contract 2 (core/contracts/bsdf.h), its shader side run on the GPU
// (tests/gpu/kernels/bsdf_probe.metal), checked against what the contract
// says:
//
//   - consistency: pdf() and evaluate() asked of a sampled direction give
//     what sample() returned with it;
//   - sampling follows pdf(): a million samples, binned over the sphere,
//     fall into each bin as often as pdf() integrated over it predicts, and
//     pdf() integrates to the fraction of samples that are not refused;
//   - the weight value |cos| / pdf, each channel's: exactly the albedo for
//     Lambert, never above 1 for a metal of f0 = 1, and its mean the
//     directional albedo, the integral of evaluate() times the cosine;
//   - a metal's Fresnel: a colored f0 against Schlick's, in double, at every
//     angle of the lobe;
//   - glass: reflection with Fresnel's probability, the exact one in double,
//     at an angle from outside and from inside; Snell's direction; weights 1
//     and 1 / eta_t^2; total internal reflection past the critical angle;
//     nothing for evaluate() and pdf();
//   - resolve(): each material kind to its Bsdf, a texture read at the
//     point, the scene's arrays read through its block.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "core/materials/coated.h"
#include "core/scene/scene.h"
#include "gpu/kernels/probes.h"
#include "gpu/support/probe_runner.h"
#include "metal/scene/scene_buffers.h"
#include "support/references.h"
#include "support/vector.h"

using namespace serenity;
using contracts::Bsdf;
using contracts::BsdfKind;
using tests::Binding;
using tests::BsdfCell;
using tests::BsdfDraw;
using tests::ProbeRunner;
using tests::Vec3;
using tests::unit;

namespace {

// What a Bsdf of a kind holds besides its color, each 0 unless given:
// named, since both are floats (I.24).
struct Parameters {
    float alpha = 0.0f;  // the conductor's
    float ior = 0.0f;    // the dielectric's and the coat's
};

Bsdf make(BsdfKind kind, Vec3 normal, contracts::Float3 color, Parameters p) {
    Bsdf b{};
    b.normal = {static_cast<float>(normal.x), static_cast<float>(normal.y), static_cast<float>(normal.z)};
    b.kind = kind;
    b.color = color;
    b.alpha = p.alpha;
    b.ior = p.ior;
    return b;
}

contracts::Float3 packed(Vec3 v) {
    return {static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
}

constexpr std::uint32_t sample_count = 1u << 20;
constexpr std::uint32_t grid = 512;  // bsdf_density's cells: grid x grid
constexpr std::uint32_t cell_count = grid * grid;

std::vector<BsdfDraw> samples(ProbeRunner& gpu, const Bsdf& bsdf, Vec3 wo) {
    return gpu.run<BsdfDraw>("bsdf_samples", sample_count,
                             {Binding::of(bsdf), Binding::of(tests::BsdfQuery{packed(wo), sample_count})});
}

std::vector<BsdfCell> density(ProbeRunner& gpu, const Bsdf& bsdf, Vec3 wo) {
    return gpu.run<BsdfCell>("bsdf_density", cell_count,
                             {Binding::of(bsdf), Binding::of(tests::BsdfQuery{packed(wo), grid})});
}

// The solid angle of each of bsdf_density's cells: the grid's are equal.
constexpr double cell_solid_angle = 4.0 * std::numbers::pi / cell_count;

// The direction at the middle of bsdf_density's cell `i`, as the kernel
// computes it (kernels/bsdf_probe.metal), in double.
Vec3 cell_direction(std::uint32_t i) {
    const double z = 1.0 - 2.0 * (i / grid + 0.5) / grid;
    const double phi = 2.0 * std::numbers::pi * (i % grid + 0.5) / grid;
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    return {r * std::cos(phi), r * std::sin(phi), z};
}

// The directional albedo, the integral of evaluate() |n . wi| over the
// sphere, each channel's, from bsdf_density's cells.
std::array<double, 3> directional_albedo(const std::vector<BsdfCell>& cells, Vec3 n) {
    std::array<double, 3> albedo{};
    for (std::uint32_t i = 0; i < cell_count; ++i) {
        const double cos_wi = std::abs(dot(cell_direction(i), n));
        for (int c = 0; c < 3; ++c) {
            albedo[static_cast<std::size_t>(c)] +=
                contracts::component(cells[i].evaluated, c) * cos_wi * cell_solid_angle;
        }
    }
    return albedo;
}

// A sample's weight, value |n . wi| / pdf, each channel's (contract 2).
std::array<double, 3> weight(const BsdfDraw& p, Vec3 n) {
    const double cos_wi = std::abs(dot(tests::vec(p.direction), n));
    return {p.value.x * cos_wi / p.pdf, p.value.y * cos_wi / p.pdf, p.value.z * cos_wi / p.pdf};
}

// Whether every channel of a sample's weight is within `tolerance` of
// `expected`'s.
bool weighs(const BsdfDraw& p, Vec3 n, const std::array<double, 3>& expected, double tolerance) {
    const std::array<double, 3> w = weight(p, n);
    return std::abs(w[0] - expected[0]) < tolerance && std::abs(w[1] - expected[1]) < tolerance &&
           std::abs(w[2] - expected[2]) < tolerance;
}

// The mean weight over every draw, refused ones counting 0: the directional
// albedo, each channel's.
std::array<double, 3> mean_weight(const std::vector<BsdfDraw>& drawn, Vec3 n) {
    std::array<double, 3> sum{};
    for (const BsdfDraw& p : drawn) {
        if (p.pdf > 0.0f) {
            const std::array<double, 3> w = weight(p, n);
            for (std::size_t c = 0; c < 3; ++c) {
                sum[c] += w[c];
            }
        }
    }
    for (double& s : sum) {
        s /= static_cast<double>(drawn.size());
    }
    return sum;
}

// The cell of bsdf_density's grid a direction falls in, coarsened by
// `coarse` in each direction.
std::size_t coarse_cell(contracts::Float3 d, std::uint32_t coarse) {
    const double z = std::clamp<double>(d.z, -1.0, 1.0);
    const double phi = std::atan2(d.y, d.x);
    const double turn = (phi < 0.0 ? phi + 2.0 * std::numbers::pi : phi) / (2.0 * std::numbers::pi);
    const std::uint32_t row = std::min(grid - 1, static_cast<std::uint32_t>((1.0 - z) * 0.5 * grid));
    const std::uint32_t column = std::min(grid - 1, static_cast<std::uint32_t>(turn * grid));
    return std::size_t{row / coarse} * (grid / coarse) + column / coarse;
}

// What check_sampling_follows_pdf() drew and evaluated, for the checks after
// it (P.9: a million draws and a quarter million cells are not made twice).
struct Sampled {
    std::vector<BsdfDraw> drawn;
    std::vector<BsdfCell> cells;
};

// Every draw that was not refused gives the pdf and value evaluate() and
// pdf() give of its direction, and is of lobe `lobe`; returns how many were
// not refused.
std::uint32_t check_consistent(const std::vector<BsdfDraw>& drawn, std::uint32_t lobe) {
    std::uint32_t valid = 0;
    std::uint32_t inconsistent = 0;
    std::uint32_t wrong_lobe = 0;
    const auto close = [](float a, float b) { return std::abs(a - b) <= 1e-3f * std::max(b, 1e-6f); };
    for (const BsdfDraw& p : drawn) {
        if (p.pdf <= 0.0f) {
            continue;
        }
        ++valid;
        wrong_lobe += p.lobe == lobe ? 0u : 1u;
        const bool same_pdf = std::abs(p.evaluated_pdf - p.pdf) <= 1e-3f * p.pdf;
        const bool same_value =
            close(p.evaluated.x, p.value.x) && close(p.evaluated.y, p.value.y) && close(p.evaluated.z, p.value.z);
        inconsistent += (same_pdf && same_value) ? 0u : 1u;
    }
    CHECK(wrong_lobe == 0);
    CHECK(inconsistent == 0);
    return valid;
}

// Pearson's chi-square of the draws that were not refused, binned 16 x 16
// over the sphere, against the pdf's integral over each bin: the test pbrt
// and Mitsuba apply to their samplers. Over the bins the pdf expects at
// least 20 samples in, and the rest pooled into one more bin, so a sample
// that lands where the pdf says almost none should is counted against it,
// not dropped. Sampling that follows the pdf gives chi^2 / dof near 1,
// within a few sqrt(2 / dof); a sampler that departs from its pdf by a few
// percent over a region holding thousands of samples gives many times that.
void check_histogram(const Sampled& s) {
    constexpr std::uint32_t coarse = 32;  // 16 x 16 bins
    constexpr std::uint32_t side = grid / coarse;
    std::vector<double> expected(std::size_t{side} * side, 0.0);
    std::vector<double> counted(expected.size(), 0.0);
    for (std::uint32_t i = 0; i < cell_count; ++i) {
        expected[std::size_t{i / grid / coarse} * side + (i % grid) / coarse] +=
            s.cells[i].pdf * cell_solid_angle * sample_count;
    }
    for (const BsdfDraw& p : s.drawn) {
        if (p.pdf > 0.0f) {
            counted[coarse_cell(p.direction, coarse)] += 1.0;
        }
    }
    double chi2 = 0.0;
    std::uint32_t bins = 0;
    double pooled_expected = 0.0;
    double pooled_counted = 0.0;
    for (std::size_t b = 0; b < expected.size(); ++b) {
        if (expected[b] >= 20.0) {
            chi2 += (counted[b] - expected[b]) * (counted[b] - expected[b]) / expected[b];
            ++bins;
        } else {
            pooled_expected += expected[b];
            pooled_counted += counted[b];
        }
    }
    if (pooled_expected >= 20.0) {
        chi2 += (pooled_counted - pooled_expected) * (pooled_counted - pooled_expected) / pooled_expected;
        ++bins;
    } else {
        // Too few expected to weigh by Pearson: the pooled bins may hold no
        // more than a few times what they expect.
        INFO("pooled bins: " << pooled_counted << " samples, " << pooled_expected << " expected");
        CHECK(pooled_counted <= 3.0 * pooled_expected + 10.0);
    }
    const double dof = bins - 1.0;
    INFO("chi^2 / dof = " << chi2 / dof << " over " << bins << " bins");
    REQUIRE(bins > 10);
    CHECK(chi2 / dof < 1.0 + 5.0 * std::sqrt(2.0 / dof));
}

// The checks every non-delta kind passes: consistency, the pdf's integral,
// and the histogram of samples against it. With `beside_delta`, the surface
// also has a delta lobe, whose samples are left out: the pdf is the other
// lobe's alone (contract 2), and integrates to the share of samples that
// lobe takes. Returns what it drew and evaluated, the delta lobe's draws
// kept.
Sampled check_sampling_follows_pdf(ProbeRunner& gpu, const Bsdf& bsdf, Vec3 wo, std::uint32_t lobe,
                                   bool beside_delta = false) {
    Sampled all{samples(gpu, bsdf, wo), density(gpu, bsdf, wo)};
    Sampled s = all;
    if (beside_delta) {
        for (BsdfDraw& p : s.drawn) {
            if ((p.lobe & contracts::lobe_delta) != 0u) {
                p.pdf = 0.0f;
            }
        }
    }
    const std::uint32_t valid = check_consistent(s.drawn, lobe);
    REQUIRE(valid > sample_count / 2);

    // The pdf integrates over the sphere to the fraction of samples not
    // refused.
    double integral = 0.0;
    for (const BsdfCell& cell : s.cells) {
        integral += cell.pdf * cell_solid_angle;
    }
    const double valid_fraction = static_cast<double>(valid) / sample_count;
    INFO("pdf integrates to " << integral << ", valid fraction " << valid_fraction);
    CHECK(std::abs(integral - valid_fraction) < 0.01);

    check_histogram(s);
    return all;
}

std::string wo_text(Vec3 wo) {
    return "wo (" + std::to_string(wo.x) + ", " + std::to_string(wo.y) + ", " + std::to_string(wo.z) + ")";
}

}  // namespace

TEST_CASE("Lambert: sampling follows the pdf, and every sample's weight is the albedo") {
    ProbeRunner gpu;
    const Vec3 n = unit(0.3, 0.2, 0.9);
    const Bsdf bsdf = make(BsdfKind::lambert, n, {0.5f, 0.7f, 0.9f}, {});
    const Vec3 wo = unit(-0.4, 0.1, 0.8);
    const Sampled front =
        check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_diffuse);

    // Each channel's weight is that channel's albedo: a channel swapped, or
    // one channel's albedo used for all, fails here.
    const std::array<double, 3> albedo{0.5, 0.7, 0.9};
    CHECK(std::ranges::count_if(front.drawn, [&](const BsdfDraw& p) { return !weighs(p, n, albedo, 1e-3); }) == 0);

    // The directional albedo, the integral of f |cos|, is the albedo, each
    // channel's.
    const std::array<double, 3> integral = directional_albedo(front.cells, n);
    for (std::size_t c = 0; c < 3; ++c) {
        INFO("channel " << c);
        CHECK(integral[c] == doctest::Approx(albedo[c]).scale(0).epsilon(0.005));
    }

    // wo on the far side of the normal: it reflects there instead. Checked
    // against the normal directly, not only against evaluate() and pdf(),
    // which would agree with sample() on the wrong side too.
    const Sampled back =
        check_sampling_follows_pdf(gpu, bsdf, -wo, contracts::lobe_reflection | contracts::lobe_diffuse);
    CHECK(std::ranges::count_if(back.drawn, [&](const BsdfDraw& p) {
              return !(p.pdf > 0.0f && dot(tests::vec(p.direction), n) < 0.0);
          }) == 0);
    CHECK(std::ranges::count_if(back.drawn, [&](const BsdfDraw& p) { return !weighs(p, n, albedo, 1e-3); }) == 0);
}

TEST_CASE("conductor: sampling follows the pdf, and a metal of f0 = 1 returns no more than it receives") {
    ProbeRunner gpu;
    // The normal along x, so a sharp lobe lies where the density grid's
    // cells are finest, at its equator, not its poles.
    const Vec3 n = unit(1.0, 0.0, 0.0);
    for (const float alpha : {0.25f, 0.06f}) {
        for (const Vec3 wo : {unit(1.0, 0.1, 0.0), unit(0.6, 0.0, 0.8), unit(0.3, 0.1, 0.95)}) {
            INFO("alpha " << alpha << ", " << wo_text(wo));
            const Bsdf bsdf = make(BsdfKind::conductor, n, {1.0f, 1.0f, 1.0f}, {.alpha = alpha});
            const Sampled s =
                check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_glossy);
            double largest = 0.0;
            for (const BsdfDraw& p : s.drawn) {
                if (p.pdf > 0.0f) {
                    largest = std::max(largest, weight(p, n)[0]);
                }
            }
            CHECK(largest <= 1.0 + 1e-3);
            // The mean weight over all draws, refused ones counting 0, is the
            // directional albedo.
            CHECK(mean_weight(s.drawn, n)[0] ==
                  doctest::Approx(directional_albedo(s.cells, n)[0]).scale(0).epsilon(0.01));
        }
    }
}

TEST_CASE("conductor: a colored f0 is Schlick's Fresnel, each channel's, at every angle") {
    // At f0 = 1, Schlick's F is 1 at every angle, so the tests above cannot
    // see the Fresnel term. Brass's f0: a metal that ignored f0's color, or
    // its angle, or used one channel for all, fails here.
    ProbeRunner gpu;
    const Vec3 n = unit(1.0, 0.0, 0.0);
    const std::array<double, 3> f0{0.91, 0.78, 0.42};
    const auto schlick = [&](std::size_t c, double cos_h) {
        const double x = 1.0 - std::clamp(cos_h, 0.0, 1.0);
        return f0[c] + (1.0 - f0[c]) * x * x * x * x * x;
    };
    // Head on, 53 degrees and 80 degrees from the normal: at 80 the half
    // vectors of the lobe are near the normal, wo . h near 0.17, and F is
    // well above f0 (blue: 0.65 against 0.42). Each wo in the plane of the
    // normal and y, so its lobe lies on the density grid's equator, where its
    // cells are finest (as the test above).
    for (const Vec3 wo : {unit(1.0, 0.1, 0.0), unit(0.6, 0.8, 0.0), unit(0.17, 0.985, 0.0)}) {
        INFO(wo_text(wo));
        const Bsdf brass = make(BsdfKind::conductor, n, {0.91f, 0.78f, 0.42f}, {.alpha = 0.25f});
        const Bsdf white = make(BsdfKind::conductor, n, {1.0f, 1.0f, 1.0f}, {.alpha = 0.25f});
        const Sampled s =
            check_sampling_follows_pdf(gpu, brass, wo, contracts::lobe_reflection | contracts::lobe_glossy);

        // evaluate() of brass over evaluate() of f0 = 1 is F(wo . h), D and
        // G cancelling: against Schlick's in double, over the lobe.
        const std::vector<BsdfCell> plain = density(gpu, white, wo);
        double worst = 0.0;
        std::uint32_t cells = 0;
        for (std::uint32_t i = 0; i < cell_count; ++i) {
            if (!(plain[i].evaluated.x > 1e-3f)) {
                continue;
            }
            ++cells;
            const Vec3 h = tests::normalized(wo + cell_direction(i));
            const double cos_h = dot(wo, h);
            for (int c = 0; c < 3; ++c) {
                const double f = static_cast<double>(contracts::component(s.cells[i].evaluated, c)) /
                                 static_cast<double>(contracts::component(plain[i].evaluated, c));
                worst = std::max(worst, std::abs(f / schlick(static_cast<std::size_t>(c), cos_h) - 1.0));
            }
        }
        INFO(cells << " cells of the lobe; largest error " << worst);
        REQUIRE(cells > 100);
        CHECK(worst < 1e-4);

        // The mean weight, each channel's, is the directional albedo, each
        // channel's.
        const std::array<double, 3> mean = mean_weight(s.drawn, n);
        const std::array<double, 3> albedo = directional_albedo(s.cells, n);
        for (std::size_t c = 0; c < 3; ++c) {
            INFO("channel " << c);
            CHECK(mean[c] == doctest::Approx(albedo[c]).scale(0).epsilon(0.01));
        }
    }
}

TEST_CASE("dielectric at an angle, from outside and from inside: "
          "Fresnel's share, Snell's direction, the eta^2 weights") {
    // Head on, exact Fresnel, Schlick's and a constant F0 agree and the
    // direction of eta does not matter; at an angle none of that holds.
    // From outside at 30 and 60 degrees, and from inside at 30, within the
    // critical angle of 41.8.
    ProbeRunner gpu;
    const double ior = 1.5;
    const Vec3 n = unit(0.0, 0.0, 1.0);
    const Bsdf glass = make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, {.ior = static_cast<float>(ior)});
    struct Incidence {
        double degrees;
        bool inside;
    };
    for (const Incidence incidence : {Incidence{30.0, false}, Incidence{60.0, false}, Incidence{30.0, true}}) {
        const double theta = incidence.degrees * std::numbers::pi / 180.0;
        const double cos_i = std::cos(theta);
        const Vec3 wo{std::sin(theta), 0.0, incidence.inside ? -cos_i : cos_i};
        // The ray arrives along -wo: eta = n_from / n_to, the facing normal
        // on wo's side, eta_t the index of wi's side over wo's.
        const double eta = incidence.inside ? ior : 1.0 / ior;
        const double eta_t = 1.0 / eta;
        const Vec3 facing = incidence.inside ? -n : n;
        const double f = tests::fresnel_reflectance(cos_i, eta);
        const double cos_t = std::sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i));
        const Vec3 reflection = 2.0 * cos_i * facing - wo;
        const Vec3 refraction = eta * -wo + (eta * cos_i - cos_t) * facing;
        INFO(incidence.degrees << " degrees from " << (incidence.inside ? "inside" : "outside") << ": F " << f);

        const auto off = [](contracts::Float3 wi, Vec3 expected) { return tests::length(tests::vec(wi) - expected); };
        std::uint32_t reflected = 0;
        std::uint32_t wrong_reflection = 0;
        std::uint32_t wrong_refraction = 0;
        for (const BsdfDraw& p : samples(gpu, glass, wo)) {
            if (p.lobe == (contracts::lobe_reflection | contracts::lobe_delta)) {
                ++reflected;
                // The mirror direction, chosen with probability F, weight 1.
                wrong_reflection += off(p.direction, reflection) < 1e-5 && std::abs(p.pdf - f) < 1e-5 &&
                                            weighs(p, n, {1.0, 1.0, 1.0}, 1e-4)
                                        ? 0u
                                        : 1u;
            } else {
                // Snell's direction, with probability 1 - F, weight
                // 1 / eta_t^2: 1 / 2.25 entering, 2.25 leaving.
                const double w = 1.0 / (eta_t * eta_t);
                wrong_refraction += p.lobe == (contracts::lobe_transmission | contracts::lobe_delta) &&
                                            off(p.direction, refraction) < 1e-5 &&
                                            std::abs(p.pdf - (1.0 - f)) < 1e-5 && weighs(p, n, {w, w, w}, 1e-4 * w)
                                        ? 0u
                                        : 1u;
            }
        }
        CHECK(wrong_reflection == 0);
        CHECK(wrong_refraction == 0);
        // The share reflected is F, within 5 standard deviations of the
        // count's binomial spread.
        const double sigma = std::sqrt(f * (1.0 - f) / sample_count);
        CHECK(std::abs(static_cast<double>(reflected) / sample_count - f) < 5.0 * sigma);
    }
}

TEST_CASE("dielectric: Fresnel chooses reflection, the weights are 1 and 1 / eta^2, "
          "and past the critical angle all reflects") {
    ProbeRunner gpu;
    const Vec3 n = unit(0.0, 0.0, 1.0);
    const Bsdf bsdf = make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, {.ior = 1.5f});

    // Straight on from outside: F = ((1.5 - 1) / (1.5 + 1))^2 = 0.04.
    std::uint32_t reflected = 0;
    std::uint32_t wrong = 0;
    for (const BsdfDraw& p : samples(gpu, bsdf, n)) {
        // A delta lobe: evaluate() and pdf() have nothing for it.
        const bool delta = p.pdf > 0.0f && p.evaluated.x == 0.0f && p.evaluated_pdf == 0.0f;
        if (p.lobe == (contracts::lobe_reflection | contracts::lobe_delta)) {
            ++reflected;
            wrong += delta && std::abs(p.direction.z - 1.0f) < 1e-6f && weighs(p, n, {1.0, 1.0, 1.0}, 1e-5) ? 0u : 1u;
        } else {
            // Straight through, into the glass, weight 1 / eta^2.
            const double w = 1.0 / 2.25;
            wrong += delta && p.lobe == (contracts::lobe_transmission | contracts::lobe_delta) &&
                             std::abs(p.direction.z + 1.0f) < 1e-6f && weighs(p, n, {w, w, w}, 1e-5)
                         ? 0u
                         : 1u;
        }
    }
    CHECK(wrong == 0);
    CHECK(static_cast<double>(reflected) / sample_count == doctest::Approx(0.04).scale(0).epsilon(0.03));

    // From inside, 60 degrees from the normal: past the critical angle,
    // asin(1 / 1.5) = 41.8 degrees, so every sample reflects, weight 1, and
    // stays inside.
    const Vec3 inside = unit(std::sin(std::numbers::pi / 3), 0.0, -std::cos(std::numbers::pi / 3));
    CHECK(std::ranges::count_if(samples(gpu, bsdf, inside), [&](const BsdfDraw& p) {
              return !(p.lobe == (contracts::lobe_reflection | contracts::lobe_delta) &&
                       weighs(p, n, {1.0, 1.0, 1.0}, 1e-5) && p.direction.z < 0.0f);
          }) == 0);
}

TEST_CASE("eta: what a transmission scales radiance by, which the path tracer's roulette undoes") {
    // bsdf_eta() is eta_t, the index of wi's side over wo's, for a
    // transmission sample drawn for wo; every refraction's weight is
    // 1 / eta_t^2 (dielectric.metal.h), so weight x eta_t^2 is 1. 1 for a
    // kind that does not transmit.
    ProbeRunner gpu;
    const Vec3 n = unit(0.0, 0.0, 1.0);
    const auto eta_of = [&](const Bsdf& bsdf, Vec3 wo) {
        return gpu.run<float>("bsdf_eta_of", 1, {Binding::of(bsdf), Binding::of(tests::BsdfQuery{packed(wo), 1u})})[0];
    };
    const Bsdf glass = make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, {.ior = 1.5f});
    const Vec3 outside = unit(0.3, 0.0, 1.0);
    const Vec3 inside = unit(0.3, 0.0, -1.0);  // within the critical angle: some of it leaves
    CHECK(eta_of(glass, outside) == doctest::Approx(1.5).scale(0).epsilon(1e-6));
    CHECK(eta_of(glass, inside) == doctest::Approx(1.0 / 1.5).scale(0).epsilon(1e-6));
    for (const Vec3 wo : {outside, inside}) {
        const double eta = eta_of(glass, wo);
        std::uint32_t transmitted = 0;
        std::uint32_t wrong = 0;
        for (const BsdfDraw& p : samples(gpu, glass, wo)) {
            if ((p.lobe & contracts::lobe_transmission) == 0u) {
                continue;
            }
            ++transmitted;
            wrong += std::abs(weight(p, n)[0] * eta * eta - 1.0) < 1e-4 ? 0u : 1u;
        }
        CHECK(transmitted > sample_count / 2);
        CHECK(wrong == 0);
    }
    for (const BsdfKind kind : {BsdfKind::none, BsdfKind::lambert, BsdfKind::conductor, BsdfKind::coated}) {
        INFO("kind " << static_cast<int>(kind));
        CHECK(eta_of(make(kind, n, {0.5f, 0.5f, 0.5f}, {.alpha = 0.3f, .ior = 1.5f}), outside) == 1.0f);
    }
}

TEST_CASE("a wo in the surface's plane: glass and a coat have no sample there, and no kind's is ever infinite") {
    // |cos wo| = 0 exactly: the delta lobes' value F / |cos| would divide by
    // 0. The sample is refused instead (pdf 0); whatever arrived along it is
    // weighed by that 0 cosine anyway.
    ProbeRunner gpu;
    const Vec3 n = unit(0.0, 0.0, 1.0);
    const Vec3 in_plane = unit(1.0, 0.0, 0.0);
    for (const BsdfKind kind : {BsdfKind::lambert, BsdfKind::conductor, BsdfKind::dielectric, BsdfKind::coated}) {
        Bsdf bsdf = make(kind, n, {0.5f, 0.5f, 0.5f},
                         {.alpha = kind == BsdfKind::conductor ? 0.3f : 0.0f,
                          .ior = kind == BsdfKind::lambert || kind == BsdfKind::conductor ? 0.0f : 1.5f});
        if (kind == BsdfKind::coated) {
            bsdf.escape = static_cast<float>(materials::internal_escape(1.5));
        }
        std::uint32_t not_finite = 0;
        std::uint32_t refused = 0;
        for (const BsdfDraw& p : samples(gpu, bsdf, in_plane)) {
            const bool finite = std::isfinite(p.pdf) && std::isfinite(p.value.x) && std::isfinite(p.value.y) &&
                                std::isfinite(p.value.z);
            not_finite += finite ? 0u : 1u;
            refused += p.pdf == 0.0f ? 1u : 0u;
        }
        INFO("kind " << static_cast<int>(kind));
        CHECK(not_finite == 0);
        if (kind == BsdfKind::dielectric || kind == BsdfKind::coated) {
            CHECK(refused == sample_count);
        }
    }
}

namespace {

Bsdf coated(Vec3 n, contracts::Float3 color, float ior) {
    Bsdf b = make(BsdfKind::coated, n, color, {.ior = ior});
    b.escape = static_cast<float>(materials::internal_escape(ior));
    return b;
}

}  // namespace

TEST_CASE("coated: the base follows its pdf, the coat is chosen by its Fresnel reflectance, weight 1") {
    ProbeRunner gpu;
    const Vec3 n = unit(0.2, -0.1, 0.95);
    const Bsdf bsdf = coated(n, {0.6f, 0.3f, 0.1f}, 1.5f);
    for (const Vec3 wo : {unit(-0.4, 0.1, 0.8), unit(0.9, 0.0, 0.25)}) {
        const double cos_o = dot(wo, n);
        const double f_o = tests::fresnel_from_air(cos_o, 1.5);
        INFO("cos theta_o " << cos_o << ", F " << f_o);
        const Sampled s =
            check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_diffuse, true);
        std::uint32_t coats = 0;
        std::uint32_t wrong = 0;
        const Vec3 mirror = 2.0 * cos_o * n - wo;
        for (const BsdfDraw& p : s.drawn) {
            if ((p.lobe & contracts::lobe_delta) == 0u) {
                continue;
            }
            ++coats;
            // The mirror of wo, chosen with probability F, weight 1 in every
            // channel: the coat has no color.
            const double off = tests::length(tests::vec(p.direction) - mirror);
            wrong += (off < 1e-4 && std::abs(p.pdf - f_o) < 1e-4 && weighs(p, n, {1.0, 1.0, 1.0}, 1e-4)) ? 0u : 1u;
        }
        CHECK(wrong == 0);
        CHECK(static_cast<double>(coats) / sample_count == doctest::Approx(f_o).scale(0).epsilon(0.02));
    }
}

TEST_CASE("coated: a white base reflects all the light that arrives, at every angle; a colored one less") {
    ProbeRunner gpu;
    const Vec3 n = unit(0.0, 0.0, 1.0);
    for (const float ior : {1.3f, 1.5f, 1.8f}) {
        for (const double cos_o : {1.0, 0.7, 0.3, 0.08}) {
            const Vec3 wo = unit(std::sqrt(1.0 - cos_o * cos_o), 0.0, cos_o);
            INFO("ior " << ior << ", cos theta_o " << cos_o);
            // The white furnace: the mean of every sample's weight, the
            // directional albedo, is 1, in every channel.
            for (const double white : mean_weight(samples(gpu, coated(n, {1.0f, 1.0f, 1.0f}, ior), wo), n)) {
                CHECK(white == doctest::Approx(1.0).scale(0).epsilon(0.003));
            }
            // A base of half: the coat's F, and the base's share of the rest.
            const double f_o = tests::fresnel_from_air(cos_o, ior);
            const double f_in = materials::internal_reflectance(ior);
            // (1 - F_o)(1 - F_out) rho / (ior^2 (1 - rho F_in)), 1 - F_out =
            // ior^2 (1 - F_in) (coated.h).
            const double expected = f_o + (1.0 - f_o) * 0.5 * (1.0 - f_in) / (1.0 - 0.5 * f_in);
            for (const double half : mean_weight(samples(gpu, coated(n, {0.5f, 0.5f, 0.5f}, ior), wo), n)) {
                CHECK(half == doctest::Approx(expected).scale(0).epsilon(0.003));
            }
        }
    }
}

TEST_CASE("coated: a coat of very high ior keeps every sample finite, and a white base still reflects all") {
    // ior 1000: F_in rounds to 1 as a float; the escape, stored instead,
    // does not, so the base's denominator stays above 0 (coated.h).
    ProbeRunner gpu;
    const Vec3 n = unit(0.0, 0.0, 1.0);
    const Vec3 wo = unit(0.6, 0.0, 0.8);
    const std::vector<BsdfDraw> drawn = samples(gpu, coated(n, {1.0f, 1.0f, 1.0f}, 1000.0f), wo);
    CHECK(std::ranges::count_if(drawn, [](const BsdfDraw& p) {
              return !(std::isfinite(p.value.x) && std::isfinite(p.pdf) && std::isfinite(p.evaluated.x));
          }) == 0);
    CHECK(mean_weight(drawn, n)[0] == doctest::Approx(1.0).scale(0).epsilon(0.01));
}

TEST_CASE("lobes: each kind's, from the shader, and an estimator aims at lights only where one is not delta") {
    using namespace contracts;
    const Vec3 n = unit(0.0, 0.0, 1.0);
    const std::vector<Bsdf> bsdfs = {
        make(BsdfKind::none, n, {1.0f, 1.0f, 1.0f}, {}),
        make(BsdfKind::lambert, n, {0.5f, 0.5f, 0.5f}, {}),
        make(BsdfKind::conductor, n, {1.0f, 1.0f, 1.0f}, {.alpha = 0.25f}),
        make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, {.ior = 1.5f}),
        make(BsdfKind::coated, n, {0.5f, 0.5f, 0.5f}, {.ior = 1.5f}),
    };
    const auto count = static_cast<std::uint32_t>(bsdfs.size());
    ProbeRunner gpu;
    const auto lobes = gpu.run<std::uint32_t>("bsdf_lobe_bits", count, {Binding::of(bsdfs), Binding::of(count)});
    CHECK(lobes[0] == 0u);
    CHECK(lobes[1] == (lobe_reflection | lobe_diffuse));
    CHECK(lobes[2] == (lobe_reflection | lobe_glossy));
    CHECK(lobes[3] == (lobe_reflection | lobe_transmission | lobe_delta));
    CHECK(lobes[4] == (lobe_reflection | lobe_diffuse | lobe_delta));
    CHECK(aims_at_lights(lobes[4]));  // its color, beside its coat's mirror

    CHECK_FALSE(aims_at_lights(lobes[0]));
    CHECK(aims_at_lights(lobes[1]));
    CHECK(aims_at_lights(lobes[2]));
    CHECK_FALSE(aims_at_lights(lobes[3]));
    CHECK(aims_at_lights(lobe_reflection | lobe_delta | lobe_glossy));  // a mirror over a glossy coat
}

TEST_CASE("resolve: each material kind to its Bsdf, a texture read at the point") {
    const scene::SceneDescription description = scene::parse(R"(
[camera]
position = [0, 1, 3]
look_at = [0, 0, 0]
vertical_fov_degrees = 40
[environment]
kind = "gradient"
zenith = [0, 0, 0]
horizon = [0, 0, 0]
[textures.checks]
kind = "checker"
size = 1
a = [0.9, 0.8, 0.7]
b = [0.1, 0.2, 0.3]
[materials.a_checks]
kind = "rough"
texture = "checks"
[materials.b_brass]
kind = "conductor"
f0 = [0.91, 0.78, 0.42]
roughness = 0.5
[materials.c_glass]
kind = "dielectric"
ior = 1.5
[materials.d_glow]
kind = "emissive"
radiance = [5, 5, 5]
[textures.swirled]
kind = "swirl"
a = [1, 0, 0]
b = [0, 0, 1]
vanes = 2
twist = 0
seed = 4
[materials.e_core]
kind = "rough"
texture = "swirled"
[materials.f_porcelain]
kind = "coated"
color = [0.6, 0.05, 0.04]
ior = 1.5
[[shapes]]
kind = "sphere"
center = [0, 0, 0]
radius = 1
material = "a_checks"
)",
                                                       "resolve test");
    const auto at = [](float x, float z, std::uint32_t material) {
        contracts::SurfaceInteraction s{};
        s.position = {x, 0.0f, z};
        s.material = material;
        s.geometric_normal = {0.0f, 1.0f, 0.0f};
        s.shading_normal = {0.0f, 1.0f, 0.0f};
        s.flags = contracts::arrived_from_outside;
        return s;
    };
    // A core's surface, in the world at `world` and on its shape at `own`.
    const auto core = [&](float world, contracts::Float3 own) {
        contracts::SurfaceInteraction s = at(world, 0.0f, 4);
        s.object_position = own;
        return s;
    };
    // Materials in name order: a_checks 0, b_brass 1, c_glass 2, d_glow 3,
    // e_core 4, f_porcelain 5.
    const std::vector<contracts::SurfaceInteraction> surfaces = {
        at(0.5f, 0.5f, 0), at(1.5f, 0.5f, 0), at(0.0f, 0.0f, 1), at(0.0f, 0.0f, 2), at(0.0f, 0.0f, 3),
        core(0.0f, {1.0f, 0.0f, 0.0f}), core(7.0f, {1.0f, 0.0f, 0.0f}), core(0.0f, {0.0f, 0.0f, 1.0f}),
        at(0.0f, 0.0f, 5)};
    const auto count = static_cast<std::uint32_t>(surfaces.size());
    ProbeRunner gpu;
    const metal::SceneBuffers buffers(gpu.device(), gpu.submission(), description);
    const auto resolved = gpu.run<Bsdf>(
        "bsdf_resolve", count,
        {Binding::of(surfaces), Binding::of(count), Binding::at(buffers.block_address())});

    CHECK(resolved[0].kind == BsdfKind::lambert);
    CHECK(resolved[0].color.x == doctest::Approx(0.9f).scale(0).epsilon(1e-6));  // floor(0.5) + floor(0.5) = 0: even, a
    CHECK(resolved[1].color.x == doctest::Approx(0.1f).scale(0).epsilon(1e-6));  // floor(1.5) + floor(0.5) = 1: odd, b
    CHECK(resolved[0].normal.y == 1.0f);
    CHECK(resolved[2].kind == BsdfKind::conductor);
    CHECK(resolved[2].color.z == doctest::Approx(0.42f).scale(0).epsilon(1e-6));
    CHECK(resolved[2].alpha == doctest::Approx(0.25f).scale(0).epsilon(1e-6));
    CHECK(resolved[3].kind == BsdfKind::dielectric);
    CHECK(resolved[3].ior == doctest::Approx(1.5f).scale(0).epsilon(1e-6));
    CHECK(resolved[4].kind == BsdfKind::none);
    // The swirl reads the point on its shape, not in the world: the same
    // point on the core at two places in the world, one color; a quarter
    // turn round the core, two vanes over, the other band.
    CHECK(resolved[5].kind == BsdfKind::lambert);
    CHECK(resolved[5].color.x == resolved[6].color.x);
    CHECK(resolved[5].color.z == resolved[6].color.z);
    CHECK(std::abs(resolved[5].color.x - resolved[7].color.x) > 0.5f);
    // Coated: its color, its coat's ior, and F_in computed at load.
    CHECK(resolved[8].kind == BsdfKind::coated);
    CHECK(resolved[8].color.x == doctest::Approx(0.6f).scale(0).epsilon(1e-6));
    CHECK(resolved[8].ior == doctest::Approx(1.5f).scale(0).epsilon(1e-6));
    CHECK(resolved[8].escape ==
          doctest::Approx(static_cast<float>(materials::internal_escape(1.5))).scale(0).epsilon(1e-6));
}
