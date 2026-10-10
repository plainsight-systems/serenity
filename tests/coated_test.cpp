// The coat's internal reflectance (core/materials/coated.h), computed at
// load: the published value for glass, and the reciprocity the coated BSDF's
// energy rests on, 1 - F_out = ior^2 (1 - F_in), F_out integrated here from
// the reflectance from outside.

#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

#include <doctest/doctest.h>

#include "core/materials/coated.h"

namespace {

// The cosine-weighted hemispherical mean of the reflectance from air into
// `ior`, by the midpoint rule.
double external_reflectance(double ior) {
    constexpr int steps = 1 << 16;
    const double h = (std::numbers::pi / 2.0) / steps;
    double sum = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double theta = (i + 0.5) * h;
        const double c = std::cos(theta);
        const double eta = 1.0 / ior;
        const double cos_t = std::sqrt(1.0 - eta * eta * (1.0 - c * c));
        const double r_s = (eta * c - cos_t) / (eta * c + cos_t);
        const double r_p = (c - eta * cos_t) / (c + eta * cos_t);
        sum += 0.5 * (r_s * r_s + r_p * r_p) * 2.0 * c * std::sin(theta) * h;
    }
    return sum;
}

}  // namespace

TEST_CASE("the coat's internal reflectance: glass's published value, and reciprocity with the outside's") {
    using serenity::materials::internal_reflectance;
    // Glass, ior 1.5: some 0.596 inside, 0.092 outside.
    CHECK(internal_reflectance(1.5) == doctest::Approx(0.596).scale(0).epsilon(0.002));
    CHECK(external_reflectance(1.5) == doctest::Approx(0.092).scale(0).epsilon(0.005));
    for (const double ior : {1.05, 1.3, 1.5, 1.8, 2.4}) {
        INFO("ior " << ior);
        CHECK(1.0 - external_reflectance(ior) ==
              doctest::Approx(ior * ior * (1.0 - internal_reflectance(ior))).scale(0).epsilon(1e-5));
        CHECK(internal_reflectance(ior) > 0.0);
        CHECK(internal_reflectance(ior) < 1.0);
    }
    // Nearer air, less is reflected inside.
    CHECK(internal_reflectance(1.05) < internal_reflectance(1.5));
}

TEST_CASE("the escape stays above 0 as a float where the reflectance rounds to 1") {
    using serenity::materials::internal_escape;
    using serenity::materials::internal_reflectance;
    // A coat of ior 1000: F_in is 1 as a float, its escape is not.
    CHECK(static_cast<float>(internal_reflectance(1000.0)) == 1.0f);
    CHECK(static_cast<float>(internal_escape(1000.0)) > 0.0f);
    // And reciprocity holds there too: 1 - F_out = ior^2 (1 - F_in).
    CHECK(1.0 - external_reflectance(1000.0) == doctest::Approx(1000.0 * 1000.0 * internal_escape(1000.0)).scale(0).epsilon(1e-4));
}

TEST_CASE("a coat with no critical angle is refused, not integrated to a NaN") {
    for (double ior : {1.0, 0.5, -2.0, std::nan(""), std::numeric_limits<double>::infinity()}) {
        CHECK_THROWS_AS(serenity::materials::internal_escape(ior), std::invalid_argument);
    }
}
