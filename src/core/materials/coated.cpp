#include "core/materials/coated.h"

#include <cmath>
#include <numbers>

namespace serenity::materials {

namespace {

// The unpolarized Fresnel reflectance at a smooth boundary, cos_i the cosine
// of incidence and eta = n_from / n_to: 1 past the critical angle. The
// shaders' (metal/materials/dielectric.metal.h), in double, for load.
double fresnel(double cos_i, double eta) {
    const double sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if (sin2_t > 1.0) {
        return 1.0;
    }
    const double cos_t = std::sqrt(1.0 - sin2_t);
    const double r_s = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    const double r_p = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (r_s * r_s + r_p * r_p);
}

}  // namespace

double internal_reflectance(double ior) {
    // From the coat's side, eta = ior / 1. Past the critical angle theta_c
    // all reflects: the integral of 2 cos sin over [theta_c, pi / 2] is
    // cos^2 theta_c = 1 - 1 / ior^2, exactly. Below it the reflectance's
    // slope is infinite at theta_c, which Simpson's rule converges on
    // slowly; theta = theta_c - s^2 makes the integrand smooth in s, over
    // [0, sqrt(theta_c)], 4096 intervals.
    const double critical = std::asin(1.0 / ior);
    constexpr int intervals = 4096;
    const double end = std::sqrt(critical);
    const double h = end / intervals;
    double sum = 0.0;
    for (int i = 0; i <= intervals; ++i) {
        const double s = i * h;
        const double theta = critical - s * s;
        const double c = std::cos(theta);
        const double value = fresnel(c, ior) * 2.0 * c * std::sin(theta) * 2.0 * s;
        const double weight = (i == 0 || i == intervals) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
        sum += weight * value;
    }
    return sum * h / 3.0 + (1.0 - 1.0 / (ior * ior));
}

}  // namespace serenity::materials
