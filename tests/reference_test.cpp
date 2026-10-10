// The reference made of batches (core/measurement/reference.h): Welford's
// mean and variance against two-pass sums in long double; the same batches
// in the same order give the same bits; a refused batch changes nothing;
// what is asked too early is refused; and the floor means what it says, on
// batches drawn from a distribution whose variance is known.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/draw.h"
#include "core/contracts/linear_image.h"
#include "core/measurement/error.h"
#include "core/measurement/reference.h"
#include "support/text.h"

using serenity::contracts::LinearImage;
using serenity::frame::Extent;
using serenity::measurement::ReferenceBuilder;
using serenity::measurement::relative_epsilon;
using serenity::tests::contains;

namespace {

// `count` batches of `extent`, each value drawn from a seed of the test's
// own (draw.h: the same numbers under any standard library): `offset` plus
// up to `spread`, so a large offset over a small spread tests the update's
// stability where summing squares would cancel.
std::vector<LinearImage> batches(Extent extent, std::size_t count, double offset, double spread,
                                 std::uint64_t seed) {
    std::vector<LinearImage> made;
    const std::size_t values = std::size_t{extent.width} * extent.height * 3;
    for (std::size_t b = 0; b < count; ++b) {
        LinearImage batch{extent, std::vector<float>(values)};
        for (std::size_t i = 0; i < values; ++i) {
            batch.rgb[i] = static_cast<float>(offset + spread * serenity::animation::draw(seed, b, i));
        }
        made.push_back(std::move(batch));
    }
    return made;
}

ReferenceBuilder built(const std::vector<LinearImage>& of) {
    ReferenceBuilder builder;
    for (const LinearImage& batch : of) {
        builder.add(batch);
    }
    return builder;
}

// Two passes in long double, independent of Welford: each value's mean,
// then the sum of its squared deviations.
struct TwoPass {
    std::vector<long double> mean;
    std::vector<long double> m2;
};

TwoPass two_pass(const std::vector<LinearImage>& of) {
    const std::size_t values = of.front().rgb.size();
    TwoPass sums{std::vector<long double>(values, 0.0L), std::vector<long double>(values, 0.0L)};
    for (const LinearImage& batch : of) {
        for (std::size_t i = 0; i < values; ++i) {
            sums.mean[i] += batch.rgb[i];
        }
    }
    for (long double& mean : sums.mean) {
        mean /= static_cast<long double>(of.size());
    }
    for (const LinearImage& batch : of) {
        for (std::size_t i = 0; i < values; ++i) {
            const long double deviation = batch.rgb[i] - sums.mean[i];
            sums.m2[i] += deviation * deviation;
        }
    }
    return sums;
}

bool same_bits(const LinearImage& a, const LinearImage& b) {
    return a.extent == b.extent && a.rgb.size() == b.rgb.size() &&
           std::memcmp(a.rgb.data(), b.rgb.data(), a.rgb.size() * sizeof(float)) == 0;
}

}  // namespace

TEST_CASE("reference: the mean and both floors equal two-pass sums in long double") {
    // Offsets of 0 and of 1000 over a spread of 1: at 1000, a float's last
    // place is 2^-14, and summed squares in float would lose the variance.
    for (const double offset : {0.0, 1000.0}) {
        INFO("offset " << offset);
        const std::vector<LinearImage> of = batches(Extent{4, 3}, 7, offset, 1.0, 11);
        const ReferenceBuilder builder = built(of);
        const TwoPass expected = two_pass(of);
        const long double n = 7.0L;
        CHECK(builder.batches() == 7);

        const LinearImage mean = builder.mean();
        CHECK(mean.extent == Extent{4, 3});
        REQUIRE(mean.rgb.size() == expected.mean.size());
        long double mse_floor = 0.0L;
        long double relative_floor = 0.0L;
        for (std::size_t i = 0; i < mean.rgb.size(); ++i) {
            INFO("value " << i);
            // Rounded to float once: within half a float's last place.
            const auto exact = static_cast<double>(expected.mean[i]);
            CHECK(std::abs(double{mean.rgb[i]} - exact) <= 0.5 * std::abs(exact) * 0x1.0p-23);
            const long double var_of_mean = expected.m2[i] / (n - 1.0L) / n;
            const long double scale = expected.mean[i] + relative_epsilon;
            mse_floor += var_of_mean;
            relative_floor += var_of_mean / (scale * scale);
        }
        const auto values = static_cast<long double>(mean.rgb.size());
        CHECK(builder.mse_floor() == doctest::Approx(static_cast<double>(mse_floor / values)).scale(0).epsilon(1e-9));
        CHECK(builder.relative_mse_floor() ==
              doctest::Approx(static_cast<double>(relative_floor / values)).scale(0).epsilon(1e-9));
    }
}

TEST_CASE("reference: one batch is its own mean, and two give the closed form") {
    // Two batches of one pixel, a and b: the mean (a + b) / 2, the variance
    // (a - b)^2 / 2, and the floor, the variance over 2, (a - b)^2 / 4.
    ReferenceBuilder builder;
    builder.add(LinearImage{Extent{1, 1}, {1.0f, 2.0f, 0.0f}});
    CHECK(builder.mean().rgb == std::vector<float>{1.0f, 2.0f, 0.0f});
    builder.add(LinearImage{Extent{1, 1}, {3.0f, 2.0f, 0.5f}});
    CHECK(builder.mean().rgb == std::vector<float>{2.0f, 2.0f, 0.25f});
    // (4 / 4 + 0 + 0.25 / 4) / 3
    CHECK(builder.mse_floor() == doctest::Approx((1.0 + 0.0 + 0.0625) / 3.0).scale(0).epsilon(1e-14));
    // Each over (mean + 0.01)^2.
    const double relative = (1.0 / (2.01 * 2.01) + 0.0 + 0.0625 / (0.26 * 0.26)) / 3.0;
    CHECK(builder.relative_mse_floor() == doctest::Approx(relative).scale(0).epsilon(1e-14));
}

TEST_CASE("reference: the same batches in the same order give the same bits") {
    const std::vector<LinearImage> of = batches(Extent{5, 4}, 9, 3.0, 2.0, 23);
    const ReferenceBuilder first = built(of);
    const ReferenceBuilder again = built(of);
    CHECK(same_bits(first.mean(), again.mean()));
    CHECK(first.mse_floor() == again.mse_floor());
    CHECK(first.relative_mse_floor() == again.relative_mse_floor());
}

TEST_CASE("reference: a refused batch changes nothing, and names itself and its pixel") {
    const std::vector<LinearImage> of = batches(Extent{3, 2}, 2, 1.0, 1.0, 5);
    ReferenceBuilder builder = built(of);
    const LinearImage mean = builder.mean();
    const double mse_floor = builder.mse_floor();
    const double relative_floor = builder.relative_mse_floor();

    const auto unchanged = [&] {
        CHECK(builder.batches() == 2);
        CHECK(same_bits(builder.mean(), mean));
        CHECK(builder.mse_floor() == mse_floor);
        CHECK(builder.relative_mse_floor() == relative_floor);
    };
    const auto refusal = [&](const LinearImage& batch) {
        return serenity::tests::error_of<std::invalid_argument>([&] { builder.add(batch); });
    };

    // Another extent, and a length its extent does not hold.
    CHECK(contains(refusal(LinearImage{Extent{2, 3}, std::vector<float>(18, 1.0f)}), "where the first batch is 3 x 2"));
    unchanged();
    LinearImage short_of_one = of.front();
    short_of_one.rgb.pop_back();
    CHECK(contains(refusal(short_of_one), "17 floats"));
    unchanged();

    // A value not finite, or below 0, at pixel (1, 1) channel 2, the last
    // value: every value before it was folded in, had it not been checked
    // first.
    for (const float wrong : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity(), -1.0f,
                              -std::numeric_limits<float>::denorm_min()}) {
        INFO("value " << wrong);
        LinearImage batch = of.front();
        batch.rgb.back() = wrong;
        const std::string why = refusal(batch);
        CHECK(contains(why, "reference batch 2"));
        CHECK(contains(why, "pixel (2, 1) channel 2"));
        unchanged();
    }
    // -0 is 0: radiance, not a refusal.
    LinearImage zero = of.front();
    zero.rgb.back() = -0.0f;
    CHECK_NOTHROW(builder.add(zero));
    CHECK(builder.batches() == 3);
}

TEST_CASE("reference: a refused first batch leaves the builder empty") {
    ReferenceBuilder builder;
    CHECK_THROWS_AS(builder.add(LinearImage{Extent{0, 1}, {}}), std::invalid_argument);
    const auto past = static_cast<std::uint32_t>(serenity::contracts::max_image_side + 1);
    CHECK_THROWS_AS(builder.add(LinearImage{Extent{past, 1}, std::vector<float>(std::size_t{past} * 3)}),
                    std::invalid_argument);
    CHECK_THROWS_AS(builder.add(LinearImage{Extent{1, 1}, {1.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f}}),
                    std::invalid_argument);
    CHECK(builder.batches() == 0);
    CHECK_THROWS_WITH_AS((void)builder.mean(), doctest::Contains("no batch has been added"), std::logic_error);
    // And a good batch after them is the first, of its own extent.
    builder.add(LinearImage{Extent{1, 1}, {1.0f, 2.0f, 3.0f}});
    CHECK(builder.mean().extent == Extent{1, 1});
}

TEST_CASE("reference: the mean before any batch, and a floor of fewer than two, are refused") {
    // std::invalid_argument is a std::logic_error too: the message tells
    // the refusal asked for from a broken image's.
    const auto no_floor = doctest::Contains("a floor needs two or more");
    ReferenceBuilder builder;
    CHECK_THROWS_WITH_AS((void)builder.mean(), doctest::Contains("no batch has been added"), std::logic_error);
    CHECK_THROWS_WITH_AS((void)builder.mse_floor(), no_floor, std::logic_error);
    CHECK_THROWS_WITH_AS((void)builder.relative_mse_floor(), no_floor, std::logic_error);
    builder.add(LinearImage{Extent{1, 1}, {1.0f, 1.0f, 1.0f}});
    CHECK_NOTHROW((void)builder.mean());
    CHECK_THROWS_WITH_AS((void)builder.mse_floor(), no_floor, std::logic_error);
    CHECK_THROWS_WITH_AS((void)builder.relative_mse_floor(), no_floor, std::logic_error);
    builder.add(LinearImage{Extent{1, 1}, {1.0f, 1.0f, 1.0f}});
    CHECK(builder.mse_floor() == 0.0);
    CHECK(builder.relative_mse_floor() == 0.0);
}

TEST_CASE("reference: the floor is the squared error the mean of B batches is expected to have") {
    // Every value of every batch drawn uniformly around mu = 1 with standard
    // deviation sigma = 0.5: mu + sigma sqrt(12) (u - 1/2), u in [0, 1). The
    // mean of B = 16 batches then has variance sigma^2 / B at every value,
    // and the floor estimates it from the batches' spread alone.
    //
    // Tolerances, from the draws' own statistics over 64 x 64 x 3 = 12288
    // independent values: a sample variance of 16 uniform draws has relative
    // standard deviation sqrt(2 / 15 - 1.2 / 16) = 0.24, so the floor, their
    // mean, has 0.24 / sqrt(12288) = 0.22%; 2% is nine of them. The squared
    // error of the mean against the truth, mean of 12288 squares of a nearly
    // normal error, has relative standard deviation sqrt(2 / 12288) = 1.3%;
    // 6% is some five of them.
    constexpr double mu = 1.0;
    constexpr double sigma = 0.5;
    constexpr std::size_t count = 16;
    constexpr Extent extent{64, 64};
    const double width = sigma * std::sqrt(12.0);
    const std::vector<LinearImage> of = batches(extent, count, mu - width / 2.0, width, 0x5eed);
    const ReferenceBuilder builder = built(of);

    const double expected = sigma * sigma / static_cast<double>(count);
    CHECK(builder.mse_floor() == doctest::Approx(expected).scale(0).epsilon(0.02));
    // The relative floor divides by the batches' own mean, m, not mu: its
    // expectation, to second order in m's spread (a symmetric draw, so the
    // variance and m are uncorrelated), is expected / (mu + epsilon)^2
    // times 1 + 3 expected / (mu + epsilon)^2, here some 4.6% above the
    // first-order figure; 2% is still nine of the floor's own deviations.
    const double scale = (mu + relative_epsilon) * (mu + relative_epsilon);
    CHECK(builder.relative_mse_floor() ==
          doctest::Approx(expected / scale * (1.0 + 3.0 * expected / scale)).scale(0).epsilon(0.02));

    // And what it is a floor of: the reference's own error against the
    // truth, measured as error.h measures any image.
    const LinearImage truth{extent, std::vector<float>(std::size_t{64} * 64 * 3, static_cast<float>(mu))};
    const auto error = serenity::measurement::error_against({.image = builder.mean(), .reference = truth});
    CHECK(error.mse == doctest::Approx(builder.mse_floor()).scale(0).epsilon(0.06));
}
