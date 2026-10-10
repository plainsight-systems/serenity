// Contract 13, the linear image (core/contracts/linear_image.h): an image
// made of zeros at its extent, the GPU's RGBA floats taken as RGB, every
// extent and length refused, the check every reader makes, and how a value's
// place is named.

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/linear_image.h"
#include "support/text.h"

using serenity::contracts::checked_values;
using serenity::contracts::from_rgba;
using serenity::contracts::LinearImage;
using serenity::contracts::make_linear_image;
using serenity::contracts::max_image_side;
using serenity::contracts::value_position;
using serenity::frame::Extent;
using serenity::tests::contains;

namespace {

constexpr auto side = static_cast<std::uint32_t>(max_image_side);

}  // namespace

TEST_CASE("linear image: made at its extent, every value 0, three a pixel") {
    const auto image = make_linear_image(Extent{5, 3});
    CHECK(image.extent == Extent{5, 3});
    REQUIRE(image.rgb.size() == 5 * 3 * 3);
    for (const float value : image.rgb) {
        CHECK(value == 0.0f);
    }
    // Each side may be the most there is, alone: a 16384 x 1 image.
    CHECK(make_linear_image(Extent{side, 1}).rgb.size() == std::size_t{side} * 3);
    CHECK(make_linear_image(Extent{1, side}).rgb.size() == std::size_t{side} * 3);
}

TEST_CASE("linear image: a side of 0 or past max_image_side is refused") {
    for (const Extent wrong : {Extent{0, 1}, Extent{1, 0}, Extent{0, 0}, Extent{side + 1, 1}, Extent{1, side + 1},
                               Extent{std::numeric_limits<std::uint32_t>::max(), 1}}) {
        INFO(wrong.width << " x " << wrong.height);
        CHECK_THROWS_AS(make_linear_image(wrong), std::invalid_argument);
        CHECK_THROWS_AS(from_rgba(wrong, std::vector<float>(4, 0.0f)), std::invalid_argument);
    }
}

TEST_CASE("linear image: from_rgba keeps each pixel's first three floats, in place, and drops its count") {
    // A different value in every float of a 3 x 2 image: a pixel moved, a
    // channel swapped or the count kept shows.
    constexpr Extent extent{3, 2};
    std::vector<float> rgba(3 * 2 * 4);
    for (std::size_t i = 0; i < rgba.size(); ++i) {
        rgba[i] = static_cast<float>(i) + 0.5f;
    }
    const auto image = from_rgba(extent, rgba);
    CHECK(image.extent == extent);
    REQUIRE(image.rgb.size() == 3 * 2 * 3);
    for (std::size_t p = 0; p < 6; ++p) {
        for (std::size_t c = 0; c < 3; ++c) {
            INFO("pixel " << p << " channel " << c);
            CHECK(image.rgb[p * 3 + c] == rgba[p * 4 + c]);
        }
    }
    // Values are taken as they are, a non-finite one included: what they
    // mean is the measurement's to judge.
    std::vector<float> odd(4, std::numeric_limits<float>::infinity());
    CHECK(from_rgba(Extent{1, 1}, odd).rgb[0] == std::numeric_limits<float>::infinity());
}

TEST_CASE("linear image: from_rgba refuses a length that is not 4 x width x height") {
    for (const std::size_t wrong : {std::size_t{0}, std::size_t{3 * 2 * 3}, std::size_t{3 * 2 * 4 - 1},
                                    std::size_t{3 * 2 * 4 + 1}}) {
        INFO(wrong << " floats");
        CHECK_THROWS_AS(from_rgba(Extent{3, 2}, std::vector<float>(wrong, 0.0f)), std::invalid_argument);
    }
}

TEST_CASE("linear image: checked_values counts what a good image holds, and refuses a broken one by name") {
    const auto refusal = [](const LinearImage& image) {
        return serenity::tests::error_of<std::invalid_argument>([&] { return checked_values(image, "reader"); });
    };
    CHECK(checked_values(make_linear_image(Extent{5, 3}), "reader") == 45);
    LinearImage short_of_one = make_linear_image(Extent{5, 3});
    short_of_one.rgb.pop_back();
    CHECK(contains(refusal(short_of_one), "reader: 44 floats for a 5 x 3 image, which holds 45"));
    LinearImage one_more = make_linear_image(Extent{5, 3});
    one_more.rgb.push_back(0.0f);
    CHECK(contains(refusal(one_more), "reader: 46 floats"));
    for (const Extent wrong : {Extent{0, 3}, Extent{5, 0}, Extent{side + 1, 1}}) {
        INFO(wrong.width << " x " << wrong.height);
        const LinearImage broken{wrong, std::vector<float>(std::size_t{wrong.width} * wrong.height * 3)};
        CHECK(contains(refusal(broken), "reader: an image of"));
    }
}

TEST_CASE("linear image: a value's position, rows from the top") {
    CHECK(value_position(Extent{5, 3}, 0) == "pixel (0, 0) channel 0");
    CHECK(value_position(Extent{5, 3}, 14) == "pixel (4, 0) channel 2");
    CHECK(value_position(Extent{5, 3}, 15) == "pixel (0, 1) channel 0");
    CHECK(value_position(Extent{5, 3}, 44) == "pixel (4, 2) channel 2");
    CHECK_THROWS_AS((void)value_position(Extent{0, 3}, 0), std::invalid_argument);
}
