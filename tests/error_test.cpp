// The error of an image against the reference (core/measurement/error.h):
// MSE and relative MSE on images whose errors are known in closed form,
// pbrt-v4's MRSE on a case worked by hand, the sums in pixel order, and
// every refusal, made before any sum.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/animation/draw.h"
#include "core/contracts/linear_image.h"
#include "core/measurement/error.h"
#include "support/text.h"

using serenity::contracts::LinearImage;
using serenity::frame::Extent;
using serenity::measurement::error_against;
using serenity::measurement::Judged;
using serenity::measurement::relative_epsilon;
using serenity::tests::contains;

namespace {

LinearImage filled(Extent extent, float value) {
    return LinearImage{extent, std::vector<float>(std::size_t{extent.width} * extent.height * 3, value)};
}

// What error_against() refuses `judged` with, or "".
std::string refusal(Judged judged) {
    return serenity::tests::error_of<std::invalid_argument>([&] { return error_against(judged); });
}

}  // namespace

TEST_CASE("error: an image equal to the reference has none") {
    const LinearImage reference{Extent{2, 1}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}};
    const auto error = error_against({.image = reference, .reference = reference});
    CHECK(error.mse == 0.0);
    CHECK(error.relative_mse == 0.0);
}

TEST_CASE("error: a uniform offset, and a mixed image, in closed form") {
    // Every value 0.5 above a reference of 1: (0.5)^2, and over (1.01)^2.
    const auto offset = error_against({.image = filled(Extent{4, 3}, 1.5f), .reference = filled(Extent{4, 3}, 1.0f)});
    CHECK(offset.mse == 0.25);
    CHECK(offset.relative_mse == doctest::Approx(0.25 / (1.01 * 1.01)).scale(0).epsilon(1e-14));

    // A dark reference value weighs its error by 1 / 0.01^2: the epsilon.
    const LinearImage reference{Extent{2, 1}, {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f}};
    const LinearImage image{Extent{2, 1}, {1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f}};
    const auto mixed = error_against({.image = image, .reference = reference});
    CHECK(mixed.mse == doctest::Approx((0.0 + 1.0 + 4.0 + 0.0 + 0.0 + 0.25) / 6.0).scale(0).epsilon(1e-15));
    const double relative = (1.0 / (1.01 * 1.01) + 4.0 / (1.01 * 1.01) + 0.25 / (0.51 * 0.51)) / 6.0;
    CHECK(mixed.relative_mse == doctest::Approx(relative).scale(0).epsilon(1e-14));
    // An image's value may be below 0 (a signed estimator's); only the
    // reference's may not.
    const auto below = error_against({.image = filled(Extent{1, 1}, -1.0f), .reference = filled(Extent{1, 1}, 0.0f)});
    CHECK(below.mse == 1.0);
}

TEST_CASE("error: the relative MSE is pbrt-v4's MRSE, worked by hand") {
    // pbrt-v4's Image::MRSE: per channel, the sum over pixels of
    // (v - r)^2 / (r + 0.01)^2, over the pixel count; imgtool's figure, the
    // channels' mean. Two pixels:
    //   pixel 0  v (1.5, 0.25, 0)      r (1, 0.25, 0.5)
    //   pixel 1  v (3, 1, 0.125)       r (2, 0, 0.125)
    // red   (0.25 / 1.01^2 + 1 / 2.01^2) / 2
    // green (0 + 1 / 0.01^2) / 2
    // blue  (0.25 / 0.51^2 + 0) / 2
    // and their mean, 10001.4537614193666 / 6 = 1666.90896023656.
    const LinearImage image{Extent{2, 1}, {1.5f, 0.25f, 0.0f, 3.0f, 1.0f, 0.125f}};
    const LinearImage reference{Extent{2, 1}, {1.0f, 0.25f, 0.5f, 2.0f, 0.0f, 0.125f}};
    const auto error = error_against({.image = image, .reference = reference});
    CHECK(error.relative_mse == doctest::Approx(1666.9089602365611).scale(0).epsilon(1e-12));
    CHECK(error.mse == doctest::Approx(2.5 / 6.0).scale(0).epsilon(1e-15));

    // And pbrt's per-channel arrangement of the same sum, written out.
    double channels = 0.0;
    for (std::size_t c = 0; c < 3; ++c) {
        double sum = 0.0;
        for (std::size_t p = 0; p < 2; ++p) {
            const double v = image.rgb[p * 3 + c];
            const double r = reference.rgb[p * 3 + c];
            sum += (v - r) * (v - r) / ((r + 0.01) * (r + 0.01));
        }
        channels += sum / 2.0;
    }
    CHECK(error.relative_mse == doctest::Approx(channels / 3.0).scale(0).epsilon(1e-12));
}

TEST_CASE("error: summed in pixel order, rows from the top: the same bits every time") {
    // Values over twelve decades, so a sum in another order rounds
    // differently; the sum in pixel order, written out here, is matched bit
    // for bit, and again.
    constexpr Extent extent{37, 23};
    LinearImage image = filled(extent, 0.0f);
    LinearImage reference = filled(extent, 0.0f);
    for (std::size_t i = 0; i < image.rgb.size(); ++i) {
        const double decade = std::floor(12.0 * serenity::animation::draw(3, i));
        image.rgb[i] = static_cast<float>(std::pow(10.0, decade - 6.0) * serenity::animation::draw(4, i));
        reference.rgb[i] = static_cast<float>(std::pow(10.0, decade - 6.0) * serenity::animation::draw(5, i));
    }
    double squared = 0.0;
    double relative = 0.0;
    for (std::size_t i = 0; i < image.rgb.size(); ++i) {
        const double d = double{image.rgb[i]} - double{reference.rgb[i]};
        const double s = double{reference.rgb[i]} + relative_epsilon;
        squared += d * d;
        relative += d * d / (s * s);
    }
    const auto values = static_cast<double>(image.rgb.size());
    const auto first = error_against({.image = image, .reference = reference});
    CHECK(first.mse == squared / values);
    CHECK(first.relative_mse == relative / values);
    const auto again = error_against({.image = image, .reference = reference});
    CHECK(again.mse == first.mse);
    CHECK(again.relative_mse == first.relative_mse);
}

TEST_CASE("error: images of two extents, or a contract broken, are refused") {
    CHECK(contains(refusal({.image = filled(Extent{2, 3}, 1.0f), .reference = filled(Extent{3, 2}, 1.0f)}),
                   "the image is 2 x 3, the reference 3 x 2"));
    LinearImage short_of_one = filled(Extent{2, 2}, 1.0f);
    short_of_one.rgb.pop_back();
    CHECK(contains(refusal({.image = short_of_one, .reference = filled(Extent{2, 2}, 1.0f)}),
                   "error_against: the image: 11 floats"));
    CHECK(contains(refusal({.image = filled(Extent{2, 2}, 1.0f), .reference = short_of_one}),
                   "error_against: the reference: 11 floats"));
    const LinearImage empty{Extent{0, 1}, {}};
    CHECK(contains(refusal({.image = empty, .reference = empty}),
                   "error_against: the reference: an image of 0 x 1, where each side is from 1"));
}

TEST_CASE("error: a value not finite in either image, or a reference value below 0, is refused by its pixel") {
    constexpr Extent extent{3, 2};
    const LinearImage ones = filled(extent, 1.0f);
    // The last value: pixel (2, 1), channel 2.
    for (const float wrong : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        INFO("value " << wrong);
        LinearImage image = ones;
        image.rgb.back() = wrong;
        CHECK(contains(refusal({.image = image, .reference = ones}), "the image's pixel (2, 1) channel 2"));
        CHECK(contains(refusal({.image = ones, .reference = image}), "the reference's pixel (2, 1) channel 2"));
    }
    LinearImage negative = ones;
    negative.rgb[4] = -0.25f;  // pixel (1, 0), channel 1
    CHECK(contains(refusal({.image = ones, .reference = negative}), "the reference's pixel (1, 0) channel 1 is -0.25"));
    // No pixel skipped: pbrt's MRSE leaves out an infinite relative error;
    // here none can arise, and a dark reference value counts in full.
    CHECK(error_against({.image = filled(extent, 1.0f), .reference = filled(extent, 0.0f)}).relative_mse ==
          doctest::Approx(1.0 / (relative_epsilon * relative_epsilon)).scale(0).epsilon(1e-12));
}
