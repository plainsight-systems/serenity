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
//   - resolve(): each material kind to its Bsdf, a texture read at the point;
//   - the numbers a path draws: a stream per pixel and frame, none shared.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numbers>
#include <span>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/bsdf.h"
#include "core/contracts/surface_interaction.h"
#include "core/materials/coated.h"
#include "core/scene/scene.h"
#include "metal/device/device.h"
#include "metal/device/library.h"
#include "metal/device/submission.h"
#include "kernels/bsdf_probe.h"
#include "serenity/metallib/smoke.h"

using namespace serenity;
using contracts::Bsdf;
using contracts::BsdfKind;

namespace {

// Runs one kernel of the test library over `threads` threads, buffer i of
// `inputs` bound at i + 1 and the output at 0, and returns the output.
class Gpu {
public:
    template <typename Out>
    std::vector<Out> run(const char* kernel, std::size_t threads, std::initializer_list<std::span<const std::byte>> inputs) {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
        MTL::Device* mtl = device_.handle();
        auto pipeline = library_.compute_pipeline(kernel);
        std::vector<NS::SharedPtr<MTL::Buffer>> buffers;
        buffers.push_back(NS::TransferPtr(mtl->newBuffer(threads * sizeof(Out), MTL::ResourceStorageModeShared)));
        for (std::span<const std::byte> input : inputs) {
            auto buffer = NS::TransferPtr(mtl->newBuffer(std::max<std::size_t>(input.size(), 16),
                                                         MTL::ResourceStorageModeShared));
            std::memcpy(buffer->contents(), input.data(), input.size());
            buffers.push_back(buffer);
        }
        auto descriptor = NS::TransferPtr(MTL4::ArgumentTableDescriptor::alloc()->init());
        descriptor->setMaxBufferBindCount(buffers.size());
        NS::Error* error = nullptr;
        auto table = NS::TransferPtr(mtl->newArgumentTable(descriptor.get(), &error));
        REQUIRE(table);
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            REQUIRE(buffers[i]);
            submission_.make_resident(buffers[i].get());
            table->setAddress(buffers[i]->gpuAddress(), i);
        }
        const auto frame = submission_.begin();
        MTL4::ComputeCommandEncoder* encoder = frame.commands->computeCommandEncoder();
        encoder->setArgumentTable(table.get());
        encoder->setComputePipelineState(pipeline.get());
        encoder->dispatchThreads(MTL::Size(threads, 1, 1), MTL::Size(pipeline->threadExecutionWidth(), 1, 1));
        encoder->endEncoding();
        submission_.commit();
        (void)submission_.wait_until_complete(frame.sequence);
        std::vector<Out> out(threads);
        std::memcpy(out.data(), buffers[0]->contents(), threads * sizeof(Out));
        return out;
    }

private:
    metal::Device device_;
    metal::Submission submission_{device_};
    metal::Library library_{device_, metallib::smoke};
};

template <typename T>
std::span<const std::byte> bytes(const T& value) {
    return std::as_bytes(std::span(&value, 1));
}

template <typename T>
std::span<const std::byte> bytes(const std::vector<T>& values) {
    return std::as_bytes(std::span(values));
}

struct V3 {
    double x, y, z;
};
double dot(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
V3 unit(double x, double y, double z) {
    const double l = std::sqrt(x * x + y * y + z * z);
    return {x / l, y / l, z / l};
}

using tests::Probe;

Bsdf make(BsdfKind kind, V3 normal, std::array<float, 3> color, float alpha, float ior) {
    Bsdf b{};
    b.normal = {float(normal.x), float(normal.y), float(normal.z)};
    b.kind = kind;
    b.color = {color[0], color[1], color[2]};
    b.alpha = alpha;
    b.ior = ior;
    return b;
}

constexpr std::uint32_t sample_count = 1u << 20;
constexpr std::uint32_t grid = 512;  // bsdf_density's cells: grid x grid

std::vector<Probe> samples(Gpu& gpu, const Bsdf& bsdf, V3 wo) {
    const float wo_count[4] = {float(wo.x), float(wo.y), float(wo.z), float(sample_count)};
    return gpu.run<Probe>("bsdf_samples", sample_count, {bytes(bsdf), bytes(wo_count)});
}

std::vector<std::array<float, 4>> density(Gpu& gpu, const Bsdf& bsdf, V3 wo) {
    const float wo_res[4] = {float(wo.x), float(wo.y), float(wo.z), float(grid)};
    return gpu.run<std::array<float, 4>>("bsdf_density", std::size_t{grid} * grid, {bytes(bsdf), bytes(wo_res)});
}

// The solid angle of each of bsdf_density's cells: the grid's are equal.
constexpr double cell_solid_angle = 4.0 * std::numbers::pi / (double(grid) * grid);

// The direction at the middle of bsdf_density's cell `i`, as the kernel
// computes it (kernels/bsdf_probe.metal), in double.
V3 cell_direction(std::uint32_t i) {
    const double z = 1.0 - 2.0 * ((i / grid) + 0.5) / grid;
    const double phi = 2.0 * std::numbers::pi * ((i % grid) + 0.5) / grid;
    const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
    return {r * std::cos(phi), r * std::sin(phi), z};
}

// The directional albedo, the integral of evaluate() |n . wi| over the
// sphere, each channel's, from bsdf_density's cells.
std::array<double, 3> directional_albedo(const std::vector<std::array<float, 4>>& cells, V3 n) {
    std::array<double, 3> albedo{};
    for (std::uint32_t i = 0; i < grid * grid; ++i) {
        const double cos = std::abs(dot(cell_direction(i), n));
        for (std::size_t c = 0; c < 3; ++c) {
            albedo[c] += cells[i][c] * cos * cell_solid_angle;
        }
    }
    return albedo;
}

V3 direction_of(const Probe& p) {
    return {p.direction.x, p.direction.y, p.direction.z};
}

// A sample's weight, value |n . wi| / pdf, each channel's (contract 2).
std::array<double, 3> weight(const Probe& p, V3 n) {
    const double cos = std::abs(dot(direction_of(p), n));
    return {p.value.x * cos / p.pdf, p.value.y * cos / p.pdf, p.value.z * cos / p.pdf};
}

// The mean weight over every draw, refused ones counting 0: the directional
// albedo, each channel's.
std::array<double, 3> mean_weight(const std::vector<Probe>& drawn, V3 n) {
    std::array<double, 3> sum{};
    for (const Probe& p : drawn) {
        if (p.pdf > 0.0f) {
            const std::array<double, 3> w = weight(p, n);
            for (std::size_t c = 0; c < 3; ++c) {
                sum[c] += w[c];
            }
        }
    }
    for (double& s : sum) {
        s /= double(drawn.size());
    }
    return sum;
}

// The unpolarized Fresnel reflectance of a smooth boundary, in double, for
// light arriving at cos_i, eta = n_from / n_to; 1 past the critical angle
// (metal/materials/fresnel.metal.h's equations).
double fresnel_reflectance(double cos_i, double eta) {
    const double sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if (sin2_t >= 1.0) {
        return 1.0;
    }
    const double cos_t = std::sqrt(1.0 - sin2_t);
    const double r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const double r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

// The cell of bsdf_density's grid a direction falls in, coarsened by
// `coarse` in each direction.
std::size_t coarse_cell(contracts::Float3 d, std::uint32_t coarse) {
    const double z = std::clamp<double>(d.z, -1.0, 1.0);
    const double phi = std::atan2(d.y, d.x);
    const auto row = std::min<std::uint32_t>(grid - 1, std::uint32_t((1.0 - z) * 0.5 * grid));
    const auto column =
        std::min<std::uint32_t>(grid - 1, std::uint32_t((phi < 0 ? phi + 2 * std::numbers::pi : phi) /
                                                         (2 * std::numbers::pi) * grid));
    return std::size_t(row / coarse) * (grid / coarse) + column / coarse;
}

// The checks every non-delta kind passes: consistency, the pdf's integral,
// and the histogram of samples against it.
// With `beside_delta`, the surface also has a delta lobe, whose samples are
// left out: the pdf is the other lobe's alone (contract 2), and integrates
// to the share of samples that lobe takes.
void check_sampling_follows_pdf(Gpu& gpu, const Bsdf& bsdf, V3 wo, std::uint32_t lobe, bool beside_delta = false) {
    auto drawn = samples(gpu, bsdf, wo);
    if (beside_delta) {
        for (Probe& p : drawn) {
            if ((p.lobe & contracts::lobe_delta) != 0u) {
                p.pdf = 0.0f;
            }
        }
    }
    const auto cells = density(gpu, bsdf, wo);

    std::uint32_t valid = 0;
    std::uint32_t inconsistent = 0;
    for (const Probe& p : drawn) {
        if (p.pdf <= 0.0f) {
            continue;
        }
        ++valid;
        CHECK(std::uint32_t(p.lobe) == lobe);
        const bool same_pdf = std::abs(p.evaluated_pdf - p.pdf) <= 1e-3f * p.pdf;
        const auto close = [](float a, float b) { return std::abs(a - b) <= 1e-3f * std::max(b, 1e-6f); };
        const bool same_value =
            close(p.evaluated.x, p.value.x) && close(p.evaluated.y, p.value.y) && close(p.evaluated.z, p.value.z);
        inconsistent += (same_pdf && same_value) ? 0u : 1u;
    }
    CHECK(inconsistent == 0);
    REQUIRE(valid > sample_count / 2);

    // The pdf integrates over the sphere to the fraction of samples not
    // refused.
    double integral = 0.0;
    for (const auto& cell : cells) {
        integral += cell[3] * cell_solid_angle;
    }
    INFO("pdf integrates to " << integral << ", valid fraction " << double(valid) / sample_count);
    CHECK(std::abs(integral - double(valid) / sample_count) < 0.01);

    // Samples per coarse bin against the pdf's integral over it.
    constexpr std::uint32_t coarse = 32;  // 16 x 16 bins
    std::vector<double> expected((grid / coarse) * (grid / coarse), 0.0);
    std::vector<double> counted(expected.size(), 0.0);
    for (std::uint32_t i = 0; i < grid * grid; ++i) {
        expected[std::size_t(i / grid / coarse) * (grid / coarse) + (i % grid) / coarse] +=
            cells[i][3] * cell_solid_angle * sample_count;
    }
    for (const Probe& p : drawn) {
        if (p.pdf > 0.0f) {
            counted[coarse_cell(p.direction, coarse)] += 1.0;
        }
    }
    // Pearson's chi-square, the test pbrt and Mitsuba apply to their
    // samplers, over the bins the pdf expects at least 20 samples in, and
    // the rest pooled into one more bin, so a sample that lands where the
    // pdf says almost none should is counted against it, not dropped.
    // Sampling that follows the pdf gives chi^2 / dof near 1, within a few
    // sqrt(2 / dof); a sampler that departs from its pdf by a few percent
    // over a region holding thousands of samples gives many times that.
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

}  // namespace

TEST_CASE("Lambert: sampling follows the pdf, and every sample's weight is the albedo") {
    Gpu gpu;
    const V3 n = unit(0.3, 0.2, 0.9);
    const Bsdf bsdf = make(BsdfKind::lambert, n, {0.5f, 0.7f, 0.9f}, 0.0f, 0.0f);
    const V3 wo = unit(-0.4, 0.1, 0.8);
    check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_diffuse);

    // Each channel's weight is that channel's albedo: a channel swapped, or
    // one channel's albedo used for all, fails here.
    const std::array<double, 3> albedo{0.5, 0.7, 0.9};
    std::uint32_t wrong_weight = 0;
    for (const Probe& p : samples(gpu, bsdf, wo)) {
        const std::array<double, 3> w = weight(p, n);
        for (std::size_t c = 0; c < 3; ++c) {
            wrong_weight += std::abs(w[c] - albedo[c]) < 1e-3 ? 0u : 1u;
        }
    }
    CHECK(wrong_weight == 0);

    // The directional albedo, the integral of f |cos|, is the albedo, each
    // channel's.
    const std::array<double, 3> integral = directional_albedo(density(gpu, bsdf, wo), n);
    for (std::size_t c = 0; c < 3; ++c) {
        INFO("channel " << c);
        CHECK(integral[c] == doctest::Approx(albedo[c]).scale(0).epsilon(0.005));
    }

    // wo on the far side of the normal: it reflects there instead. Checked
    // against the normal directly, not only against evaluate() and pdf(),
    // which would agree with sample() on the wrong side too.
    const V3 behind{-wo.x, -wo.y, -wo.z};
    check_sampling_follows_pdf(gpu, bsdf, behind, contracts::lobe_reflection | contracts::lobe_diffuse);
    std::uint32_t wrong_side = 0;
    std::uint32_t wrong_back_weight = 0;
    for (const Probe& p : samples(gpu, bsdf, behind)) {
        wrong_side += (p.pdf > 0.0f && dot(direction_of(p), n) < 0.0) ? 0u : 1u;
        const std::array<double, 3> w = weight(p, n);
        for (std::size_t c = 0; c < 3; ++c) {
            wrong_back_weight += std::abs(w[c] - albedo[c]) < 1e-3 ? 0u : 1u;
        }
    }
    CHECK(wrong_side == 0);
    CHECK(wrong_back_weight == 0);
}

TEST_CASE("conductor: sampling follows the pdf, and a metal of f0 = 1 returns no more than it receives") {
    Gpu gpu;
    // The normal along x, so a sharp lobe lies where the density grid's
    // cells are finest, at its equator, not its poles.
    const V3 n = unit(1.0, 0.0, 0.0);
    for (float alpha : {0.25f, 0.06f}) {
        for (V3 wo : {unit(1.0, 0.1, 0.0), unit(0.6, 0.0, 0.8), unit(0.3, 0.1, 0.95)}) {
            INFO("alpha " << alpha << ", wo (" << wo.x << ", " << wo.y << ", " << wo.z << ")");
            const Bsdf bsdf = make(BsdfKind::conductor, n, {1.0f, 1.0f, 1.0f}, alpha, 0.0f);
            check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_glossy);

            const std::vector<Probe> drawn = samples(gpu, bsdf, wo);
            double largest = 0.0;
            for (const Probe& p : drawn) {
                if (p.pdf > 0.0f) {
                    largest = std::max(largest, weight(p, n)[0]);
                }
            }
            CHECK(largest <= 1.0 + 1e-3);
            // The mean weight over all draws, refused ones counting 0, is the
            // directional albedo.
            CHECK(mean_weight(drawn, n)[0] ==
                  doctest::Approx(directional_albedo(density(gpu, bsdf, wo), n)[0]).scale(0).epsilon(0.01));
        }
    }
}

TEST_CASE("conductor: a colored f0 is Schlick's Fresnel, each channel's, at every angle") {
    // At f0 = 1, Schlick's F is 1 at every angle, so the tests above cannot
    // see the Fresnel term. Brass's f0: a metal that ignored f0's color, or
    // its angle, or used one channel for all, fails here.
    Gpu gpu;
    const V3 n = unit(1.0, 0.0, 0.0);
    const std::array<double, 3> f0{0.91, 0.78, 0.42};
    const auto schlick = [&](std::size_t c, double cos) {
        const double x = 1.0 - std::clamp(cos, 0.0, 1.0);
        return f0[c] + (1.0 - f0[c]) * x * x * x * x * x;
    };
    // Head on, 53 degrees and 80 degrees from the normal: at 80 the half
    // vectors of the lobe are near the normal, wo . h near 0.17, and F is
    // well above f0 (blue: 0.65 against 0.42). Each wo in the plane of the
    // normal and y, so its lobe lies on the density grid's equator, where its
    // cells are finest (as the test above).
    for (const V3 wo : {unit(1.0, 0.1, 0.0), unit(0.6, 0.8, 0.0), unit(0.17, 0.985, 0.0)}) {
        INFO("wo (" << wo.x << ", " << wo.y << ", " << wo.z << ")");
        const Bsdf brass = make(BsdfKind::conductor, n, {0.91f, 0.78f, 0.42f}, 0.25f, 0.0f);
        const Bsdf white = make(BsdfKind::conductor, n, {1.0f, 1.0f, 1.0f}, 0.25f, 0.0f);
        check_sampling_follows_pdf(gpu, brass, wo, contracts::lobe_reflection | contracts::lobe_glossy);

        // evaluate() of brass over evaluate() of f0 = 1 is F(wo . h), D and
        // G cancelling: against Schlick's in double, over the lobe.
        const auto colored = density(gpu, brass, wo);
        const auto plain = density(gpu, white, wo);
        double worst = 0.0;
        double farthest_from_f0 = 0.0;
        std::uint32_t cells = 0;
        for (std::uint32_t i = 0; i < grid * grid; ++i) {
            if (!(plain[i][0] > 1e-3f)) {
                continue;
            }
            ++cells;
            const V3 wi = cell_direction(i);
            const V3 h = unit(wo.x + wi.x, wo.y + wi.y, wo.z + wi.z);
            const double cos = dot(wo, h);
            for (std::size_t c = 0; c < 3; ++c) {
                const double f = double(colored[i][c]) / double(plain[i][c]);
                worst = std::max(worst, std::abs(f / schlick(c, cos) - 1.0));
                farthest_from_f0 = std::max(farthest_from_f0, schlick(c, cos) - f0[c]);
            }
        }
        INFO(cells << " cells of the lobe; largest error " << worst << "; F at most " << farthest_from_f0
                   << " above f0");
        REQUIRE(cells > 100);
        CHECK(worst < 1e-4);

        // The mean weight, each channel's, is the directional albedo, each
        // channel's.
        const std::array<double, 3> mean = mean_weight(samples(gpu, brass, wo), n);
        const std::array<double, 3> albedo = directional_albedo(colored, n);
        for (std::size_t c = 0; c < 3; ++c) {
            INFO("channel " << c);
            CHECK(mean[c] == doctest::Approx(albedo[c]).scale(0).epsilon(0.01));
        }
    }
}

TEST_CASE("dielectric at an angle, from outside and from inside: Fresnel's share, Snell's direction, the eta^2 weights") {
    // Head on, exact Fresnel, Schlick's and a constant F0 agree and the
    // direction of eta does not matter; at an angle none of that holds.
    // From outside at 30 and 60 degrees, and from inside at 30, within the
    // critical angle of 41.8.
    Gpu gpu;
    const double ior = 1.5;
    const V3 n = unit(0.0, 0.0, 1.0);
    const Bsdf glass = make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, 0.0f, float(ior));
    struct Incidence {
        double degrees;
        bool inside;
    };
    for (const auto [degrees, inside] : {Incidence{30.0, false}, Incidence{60.0, false}, Incidence{30.0, true}}) {
        const double theta = degrees * std::numbers::pi / 180.0;
        const double cos_i = std::cos(theta);
        const V3 wo{std::sin(theta), 0.0, inside ? -cos_i : cos_i};
        // The ray arrives along -wo: eta = n_from / n_to, the facing normal
        // on wo's side, eta_t the index of wi's side over wo's.
        const double eta = inside ? ior : 1.0 / ior;
        const double eta_t = 1.0 / eta;
        const V3 facing{0.0, 0.0, inside ? -1.0 : 1.0};
        const double f = fresnel_reflectance(cos_i, eta);
        const double cos_t = std::sqrt(1.0 - eta * eta * (1.0 - cos_i * cos_i));
        const V3 reflection{2.0 * cos_i * facing.x - wo.x, 2.0 * cos_i * facing.y - wo.y,
                            2.0 * cos_i * facing.z - wo.z};
        const V3 refraction{-eta * wo.x + (eta * cos_i - cos_t) * facing.x,
                            -eta * wo.y + (eta * cos_i - cos_t) * facing.y,
                            -eta * wo.z + (eta * cos_i - cos_t) * facing.z};
        const std::string side = inside ? "inside" : "outside";
        INFO(degrees << " degrees from " << side << ": F " << f);

        std::uint32_t reflected = 0;
        std::uint32_t wrong_reflection = 0;
        std::uint32_t wrong_refraction = 0;
        for (const Probe& p : samples(gpu, glass, wo)) {
            const V3 wi = direction_of(p);
            const std::array<double, 3> w = weight(p, n);
            const auto off = [&](V3 expected) {
                return std::abs(wi.x - expected.x) + std::abs(wi.y - expected.y) + std::abs(wi.z - expected.z);
            };
            const auto all = [&](double expected) {
                return std::abs(w[0] - expected) < 1e-4 * expected && std::abs(w[1] - expected) < 1e-4 * expected &&
                       std::abs(w[2] - expected) < 1e-4 * expected;
            };
            if (p.lobe == (contracts::lobe_reflection | contracts::lobe_delta)) {
                ++reflected;
                // The mirror direction, chosen with probability F, weight 1.
                wrong_reflection += off(reflection) < 1e-5 && std::abs(p.pdf - f) < 1e-5 && all(1.0) ? 0u : 1u;
            } else {
                // Snell's direction, with probability 1 - F, weight
                // 1 / eta_t^2: 1 / 2.25 entering, 2.25 leaving.
                wrong_refraction += p.lobe == (contracts::lobe_transmission | contracts::lobe_delta) &&
                                            off(refraction) < 1e-5 && std::abs(p.pdf - (1.0 - f)) < 1e-5 &&
                                            all(1.0 / (eta_t * eta_t))
                                        ? 0u
                                        : 1u;
            }
        }
        CHECK(wrong_reflection == 0);
        CHECK(wrong_refraction == 0);
        // The share reflected is F, within 5 standard deviations of the
        // count's binomial spread.
        const double sigma = std::sqrt(f * (1.0 - f) / sample_count);
        CHECK(std::abs(double(reflected) / sample_count - f) < 5.0 * sigma);
    }
}

TEST_CASE("dielectric: Fresnel chooses reflection, the weights are 1 and 1 / eta^2, and past the critical angle all reflects") {
    Gpu gpu;
    const V3 n = unit(0.0, 0.0, 1.0);
    const Bsdf bsdf = make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, 0.0f, 1.5f);

    // Straight on from outside: F = ((1.5 - 1) / (1.5 + 1))^2 = 0.04.
    std::uint32_t reflected = 0;
    for (const Probe& p : samples(gpu, bsdf, n)) {
        REQUIRE(p.pdf > 0.0f);
        CHECK(p.evaluated.x == 0.0f);
        CHECK(p.evaluated_pdf == 0.0f);
        const double weight = p.value.x * std::abs(p.direction.z) / p.pdf;
        if (std::uint32_t(p.lobe) == (contracts::lobe_reflection | contracts::lobe_delta)) {
            ++reflected;
            CHECK(p.direction.z == doctest::Approx(1.0f).scale(0).epsilon(1e-6));
            CHECK(weight == doctest::Approx(1.0).scale(0).epsilon(1e-6));
        } else {
            CHECK(std::uint32_t(p.lobe) == (contracts::lobe_transmission | contracts::lobe_delta));
            CHECK(p.direction.z == doctest::Approx(-1.0f).scale(0).epsilon(1e-6));  // straight through, into the glass
            CHECK(weight == doctest::Approx(1.0 / 2.25).scale(0).epsilon(1e-6));
        }
    }
    CHECK(double(reflected) / sample_count == doctest::Approx(0.04).scale(0).epsilon(0.03));

    // From inside, 60 degrees from the normal: past the critical angle,
    // asin(1 / 1.5) = 41.8 degrees, so every sample reflects, weight 1.
    const V3 inside = unit(std::sin(std::numbers::pi / 3), 0.0, -std::cos(std::numbers::pi / 3));
    for (const Probe& p : samples(gpu, bsdf, inside)) {
        CHECK(std::uint32_t(p.lobe) == (contracts::lobe_reflection | contracts::lobe_delta));
        CHECK(p.value.x * std::abs(p.direction.z) / p.pdf == doctest::Approx(1.0).scale(0).epsilon(1e-6));
        CHECK(p.direction.z < 0.0f);  // stays inside
    }
}

namespace {

// PCG's output permutation, as the shaders compute it (metal/math/
// hash.metal.h): to show which pixels the sampler's old 32-bit key merged.
std::uint32_t pcg_hash(std::uint32_t v) {
    const std::uint32_t state = v * 747796405u + 2891336453u;
    const std::uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

}  // namespace

TEST_CASE("a path's numbers: a stream per pixel and frame, none shared, each drawn again exactly") {
    // Pixels (2627, 65) and (0, 818) of frame 0: the old key, a 32-bit hash
    // of pixel and frame, was the same for both, and so were their paths'
    // numbers, all the way down (GDSA.3). The stream is now (pixel, frame)
    // itself (metal/sampler/sampler.metal.h).
    const auto old_key = [](std::uint32_t x, std::uint32_t y, std::uint32_t frame) {
        return pcg_hash(x ^ pcg_hash(y ^ pcg_hash(frame)));
    };
    REQUIRE(old_key(2627, 65, 0) == old_key(0, 818, 0));

    Gpu gpu;
    const std::vector<std::array<std::uint32_t, 4>> queries = {
        {2627, 65, 0, 0}, {0, 818, 0, 0}, {2627, 65, 0, 0}, {2627, 65, 1, 0}, {65, 2627, 0, 0}};
    const auto count = std::uint32_t(queries.size());
    const auto numbers = gpu.run<std::array<float, 8>>("path_numbers_probe", queries.size(), {bytes(queries), bytes(count)});
    for (const auto& path : numbers) {
        for (const float u : path) {
            CHECK(u >= 0.0f);
            CHECK(u < 1.0f);
        }
    }
    CHECK(numbers[0] != numbers[1]);  // the pixels the old key merged
    CHECK(numbers[0] == numbers[2]);  // the same pixel and frame, drawn again
    CHECK(numbers[0] != numbers[3]);  // the next frame
    CHECK(numbers[0] != numbers[4]);  // x and y swapped
    // Within a path, no number repeats the one before.
    for (std::size_t k = 1; k < 8; ++k) {
        CHECK(numbers[0][k] != numbers[0][k - 1]);
    }
}

TEST_CASE("eta: what a transmission scales radiance by, which the path tracer's roulette undoes") {
    // bsdf_eta() is eta_t, the index of wi's side over wo's, for a
    // transmission sample drawn for wo; every refraction's weight is
    // 1 / eta_t^2 (dielectric.metal.h), so weight x eta_t^2 is 1. 1 for a
    // kind that does not transmit.
    Gpu gpu;
    const V3 n = unit(0.0, 0.0, 1.0);
    const auto eta_of = [&](const Bsdf& bsdf, V3 wo) {
        const float wo_count[4] = {float(wo.x), float(wo.y), float(wo.z), 1.0f};
        return gpu.run<float>("bsdf_eta_of", 1, {bytes(bsdf), bytes(wo_count)})[0];
    };
    const Bsdf glass = make(BsdfKind::dielectric, n, {1.0f, 1.0f, 1.0f}, 0.0f, 1.5f);
    const V3 outside = unit(0.3, 0.0, 1.0);
    const V3 inside = unit(0.3, 0.0, -1.0);  // within the critical angle: some of it leaves
    CHECK(eta_of(glass, outside) == doctest::Approx(1.5).scale(0).epsilon(1e-6));
    CHECK(eta_of(glass, inside) == doctest::Approx(1.0 / 1.5).scale(0).epsilon(1e-6));
    for (const V3 wo : {outside, inside}) {
        const double eta = eta_of(glass, wo);
        std::uint32_t transmitted = 0;
        std::uint32_t wrong = 0;
        for (const Probe& p : samples(gpu, glass, wo)) {
            if ((p.lobe & contracts::lobe_transmission) == 0u) {
                continue;
            }
            ++transmitted;
            const double weight = p.value.x * std::abs(p.direction.z) / p.pdf;
            wrong += std::abs(weight * eta * eta - 1.0) < 1e-4 ? 0u : 1u;
        }
        CHECK(transmitted > sample_count / 2);
        CHECK(wrong == 0);
    }
    for (const BsdfKind kind : {BsdfKind::none, BsdfKind::lambert, BsdfKind::conductor, BsdfKind::coated}) {
        INFO("kind " << int(kind));
        CHECK(eta_of(make(kind, n, {0.5f, 0.5f, 0.5f}, 0.3f, 1.5f), outside) == 1.0f);
    }
}

TEST_CASE("a wo in the surface's plane: glass and a coat have no sample there, and no kind's is ever infinite") {
    // |cos wo| = 0 exactly: the delta lobes' value F / |cos| would divide by
    // 0. The sample is refused instead (pdf 0); whatever arrived along it is
    // weighed by that 0 cosine anyway.
    Gpu gpu;
    const V3 n = unit(0.0, 0.0, 1.0);
    const V3 in_plane = unit(1.0, 0.0, 0.0);
    for (const BsdfKind kind : {BsdfKind::lambert, BsdfKind::conductor, BsdfKind::dielectric, BsdfKind::coated}) {
        Bsdf bsdf = make(kind, n, {0.5f, 0.5f, 0.5f}, kind == BsdfKind::conductor ? 0.3f : 0.0f,
                         kind == BsdfKind::lambert || kind == BsdfKind::conductor ? 0.0f : 1.5f);
        if (kind == BsdfKind::coated) {
            bsdf.escape = float(materials::internal_escape(1.5));
        }
        std::uint32_t not_finite = 0;
        std::uint32_t refused = 0;
        for (const Probe& p : samples(gpu, bsdf, in_plane)) {
            not_finite += std::isfinite(p.pdf) && std::isfinite(p.value.x) && std::isfinite(p.value.y) &&
                                  std::isfinite(p.value.z)
                              ? 0u
                              : 1u;
            refused += p.pdf == 0.0f ? 1u : 0u;
        }
        INFO("kind " << int(kind));
        CHECK(not_finite == 0);
        if (kind == BsdfKind::dielectric || kind == BsdfKind::coated) {
            CHECK(refused == sample_count);
        }
    }
}

namespace {

// The unpolarized Fresnel reflectance from air into `ior`, in double: the
// coat's (materials/coated.h).
double fresnel_from_air(double cos_i, double ior) {
    return fresnel_reflectance(cos_i, 1.0 / ior);
}

Bsdf coated(V3 n, std::array<float, 3> color, float ior) {
    Bsdf b = make(BsdfKind::coated, n, color, 0.0f, ior);
    b.escape = float(materials::internal_escape(ior));
    return b;
}

}  // namespace

TEST_CASE("coated: the base follows its pdf, the coat is chosen by its Fresnel reflectance, weight 1") {
    Gpu gpu;
    const V3 n = unit(0.2, -0.1, 0.95);
    const Bsdf bsdf = coated(n, {0.6f, 0.3f, 0.1f}, 1.5f);
    for (const V3 wo : {unit(-0.4, 0.1, 0.8), unit(0.9, 0.0, 0.25)}) {
        const double cos_o = dot(wo, n);
        const double f_o = fresnel_from_air(cos_o, 1.5);
        INFO("cos theta_o " << cos_o << ", F " << f_o);
        check_sampling_follows_pdf(gpu, bsdf, wo, contracts::lobe_reflection | contracts::lobe_diffuse, true);
        std::uint32_t coats = 0;
        std::uint32_t wrong = 0;
        for (const Probe& p : samples(gpu, bsdf, wo)) {
            if ((p.lobe & contracts::lobe_delta) == 0u) {
                continue;
            }
            ++coats;
            const V3 wi = direction_of(p);
            // The mirror of wo, 2 cos n - wo, chosen with probability F,
            // weight 1 in every channel: the coat has no color.
            const double off = std::abs(wi.x - (2.0 * cos_o * n.x - wo.x)) +
                               std::abs(wi.y - (2.0 * cos_o * n.y - wo.y)) +
                               std::abs(wi.z - (2.0 * cos_o * n.z - wo.z));
            const std::array<double, 3> w = weight(p, n);
            const bool weight_one =
                std::abs(w[0] - 1.0) < 1e-4 && std::abs(w[1] - 1.0) < 1e-4 && std::abs(w[2] - 1.0) < 1e-4;
            wrong += (off < 1e-4 && std::abs(p.pdf - f_o) < 1e-4 && weight_one) ? 0u : 1u;
        }
        CHECK(wrong == 0);
        CHECK(double(coats) / sample_count == doctest::Approx(f_o).scale(0).epsilon(0.02));
    }
}

TEST_CASE("coated: a white base reflects all the light that arrives, at every angle; a colored one less") {
    Gpu gpu;
    const V3 n = unit(0.0, 0.0, 1.0);
    for (const float ior : {1.3f, 1.5f, 1.8f}) {
        for (const double cos_o : {1.0, 0.7, 0.3, 0.08}) {
            const V3 wo = unit(std::sqrt(1.0 - cos_o * cos_o), 0.0, cos_o);
            INFO("ior " << ior << ", cos theta_o " << cos_o);
            // The white furnace: the mean of every sample's weight, the
            // directional albedo, is 1, in every channel.
            for (const double white : mean_weight(samples(gpu, coated(n, {1.0f, 1.0f, 1.0f}, ior), wo), n)) {
                CHECK(white == doctest::Approx(1.0).scale(0).epsilon(0.003));
            }
            // A base of half: the coat's F, and the base's share of the rest.
            const double f_o = fresnel_from_air(cos_o, ior);
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
    Gpu gpu;
    const V3 n = unit(0.0, 0.0, 1.0);
    const V3 wo = unit(0.6, 0.0, 0.8);
    const std::vector<Probe> drawn = samples(gpu, coated(n, {1.0f, 1.0f, 1.0f}, 1000.0f), wo);
    std::uint32_t not_finite = 0;
    for (const Probe& p : drawn) {
        not_finite += std::isfinite(p.value.x) && std::isfinite(p.pdf) && std::isfinite(p.evaluated.x) ? 0u : 1u;
    }
    CHECK(not_finite == 0);
    CHECK(mean_weight(drawn, n)[0] == doctest::Approx(1.0).scale(0).epsilon(0.01));
}

TEST_CASE("lobes: each kind's, from the shader, and an estimator aims at lights only where one is not delta") {
    using namespace contracts;
    const V3 n = unit(0.0, 0.0, 1.0);
    const std::vector<Bsdf> bsdfs = {
        make(BsdfKind::none, n, {1, 1, 1}, 0.0f, 0.0f),
        make(BsdfKind::lambert, n, {0.5f, 0.5f, 0.5f}, 0.0f, 0.0f),
        make(BsdfKind::conductor, n, {1, 1, 1}, 0.25f, 0.0f),
        make(BsdfKind::dielectric, n, {1, 1, 1}, 0.0f, 1.5f),
        make(BsdfKind::coated, n, {0.5f, 0.5f, 0.5f}, 0.0f, 1.5f),
    };
    const std::uint32_t count = std::uint32_t(bsdfs.size());
    Gpu gpu;
    const auto lobes = gpu.run<std::uint32_t>("bsdf_lobe_bits", count, {bytes(bsdfs), bytes(count)});
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
    const scene::SceneDescription scene = scene::parse(R"(
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
    const std::uint32_t count = std::uint32_t(surfaces.size());
    Gpu gpu;
    const auto resolved = gpu.run<Bsdf>(
        "bsdf_resolve", count,
        {bytes(surfaces), bytes(scene.materials), bytes(scene.rough), bytes(scene.dielectrics),
         bytes(scene.conductors), bytes(scene.textures), bytes(scene.checkers), bytes(count), bytes(scene.coated),
         bytes(scene.woods), bytes(scene.swirls)});

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
    CHECK(resolved[8].escape == doctest::Approx(float(materials::internal_escape(1.5))).scale(0).epsilon(1e-6));
}
