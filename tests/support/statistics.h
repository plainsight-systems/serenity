#pragma once

// The statistics the sampling tests judge draws by, written once (ES.3,
// F.10): Pearson's chi-square of counts against what a distribution
// expects, with its p-value, and the correlation of two series.
//
// Two acceptance rules, each for its own kind of expected count:
//
//   - fits(): the p-value at `significance`, for counts expected exactly
//     (a softmax, a uniform grid): a correct sampler fails one seed in a
//     thousand, and the seeds are fixed (encoding_test.cpp);
//   - within_five_sigma(): chi^2 / dof under 1 + 5 sqrt(2 / dof), five of
//     its standard deviations, for counts expected through a quadrature of
//     the pdf over each bin (bsdf_test.cpp, light_test.cpp): at a million
//     draws a bin's midpoint rule is off by more than its noise allows a
//     p-value test, and five sigma still refuses a sampler a few percent
//     off over thousands of draws.

#include <cmath>
#include <cstddef>
#include <vector>

#include <doctest/doctest.h>

namespace serenity::tests {

// Q(a, x), the regularized upper incomplete gamma function (Numerical
// Recipes' gammq: the series below a + 1, the continued fraction above).
inline double upper_gamma(double a, double x) {
    if (x <= 0.0) {
        return 1.0;
    }
    const double log_front = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n < 1000 && std::abs(term) > std::abs(sum) * 1e-15; ++n) {
            term *= x / (a + n);
            sum += term;
        }
        return 1.0 - sum * std::exp(log_front);
    }
    constexpr double tiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 1000; ++i) {
        const double an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        d = std::abs(d) < tiny ? tiny : d;
        c = b + an / c;
        c = std::abs(c) < tiny ? tiny : c;
        d = 1.0 / d;
        const double step = d * c;
        h *= step;
        if (std::abs(step - 1.0) < 1e-15) {
            break;
        }
    }
    return std::exp(log_front) * h;
}

// Pearson's chi-square of `counted` against `expected`, its degrees of
// freedom (the bins less one), and the chance of one at least as large
// under the expected distribution.
struct ChiSquare {
    double statistic = 0.0;
    std::size_t dof = 0;
    double p = 0.0;
};

inline ChiSquare chi_square(const std::vector<double>& counted, const std::vector<double>& expected) {
    REQUIRE(counted.size() == expected.size());
    REQUIRE(counted.size() > 1);
    ChiSquare result;
    for (std::size_t i = 0; i < counted.size(); ++i) {
        result.statistic += (counted[i] - expected[i]) * (counted[i] - expected[i]) / expected[i];
    }
    result.dof = counted.size() - 1;
    result.p = upper_gamma(0.5 * static_cast<double>(result.dof), 0.5 * result.statistic);
    return result;
}

// The significance fits() judges at: a correct sampler fails one in a
// thousand seeds, and the seeds are fixed.
inline constexpr double significance = 1e-3;

// Whether counts expected exactly fit (see above).
inline bool fits(const ChiSquare& fit) {
    return fit.p > significance;
}

// Whether counts expected through a quadrature are within five standard
// deviations of chi^2 / dof's mean (see above).
inline bool within_five_sigma(const ChiSquare& fit) {
    const auto dof = static_cast<double>(fit.dof);
    return fit.statistic / dof < 1.0 + 5.0 * std::sqrt(2.0 / dof);
}

// The correlation of a[i] and b[i] over i.
inline double correlation(const std::vector<double>& a, const std::vector<double>& b) {
    REQUIRE(a.size() == b.size());
    REQUIRE_FALSE(a.empty());
    const auto n = static_cast<double>(a.size());
    double sa = 0.0;
    double sb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sa += a[i];
        sb += b[i];
    }
    const double ma = sa / n;
    const double mb = sb / n;
    double cov = 0.0;
    double va = 0.0;
    double vb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        cov += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma) * (a[i] - ma);
        vb += (b[i] - mb) * (b[i] - mb);
    }
    return cov / std::sqrt(va * vb);
}

}  // namespace serenity::tests
