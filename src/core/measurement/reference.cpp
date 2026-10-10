#include "core/measurement/reference.h"

#include <cmath>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/measurement/error.h"

namespace serenity::measurement {

void ReferenceBuilder::add(const contracts::LinearImage& batch) {
    // What a refusal names: the batch, counted from 0.
    const std::string which = "reference batch " + std::to_string(count_);
    const frame::Extent extent = batch.extent;
    // Every check before anything is folded in or allocated, so a refused
    // batch leaves the state as it was (I.5, E.4): its extent the first's,
    // contract 13's own check, then each value.
    if (count_ != 0 && !(extent == extent_)) {
        throw std::invalid_argument(which + ": " + std::to_string(extent.width) + " x " +
                                    std::to_string(extent.height) + ", where the first batch is " +
                                    std::to_string(extent_.width) + " x " + std::to_string(extent_.height));
    }
    const std::size_t values = contracts::checked_values(batch, which);
    for (std::size_t i = 0; i < values; ++i) {
        const float x = batch.rgb[i];
        if (!std::isfinite(x) || x < 0.0f) {
            throw std::invalid_argument(which + ": " + contracts::value_position(extent, i) + " is " +
                                        std::format("{}", x) + ", where radiance is finite and 0 or more");
        }
    }

    // The state, made at the first batch: built aside and moved in, so an
    // allocation that fails leaves the builder empty, as it was (E.4, MEM.9:
    // allocated once, never again).
    if (count_ == 0) {
        std::vector<double> mean(values, 0.0);
        std::vector<double> m2(values, 0.0);
        mean_ = std::move(mean);
        m2_ = std::move(m2);
        extent_ = extent;
    }

    // Welford's update, in double, in pixel order: one pass, stable where
    // summing squares cancels, and the batches folded in the order given, so
    // the same batches in the same order give the same bits (CDSA.23;
    // GDSA.2, run to run). Nothing below can throw.
    const double n = static_cast<double>(count_ + 1);  // exact: far under 2^53 batches (ES.46)
    for (std::size_t i = 0; i < values; ++i) {
        const double x = batch.rgb[i];
        const double delta = x - mean_[i];
        mean_[i] += delta / n;
        m2_[i] += delta * (x - mean_[i]);
    }
    ++count_;
}

contracts::LinearImage ReferenceBuilder::mean() const {
    if (count_ == 0) {
        throw std::logic_error("ReferenceBuilder::mean: no batch has been added");
    }
    contracts::LinearImage image = contracts::make_linear_image(extent_);
    for (std::size_t i = 0; i < mean_.size(); ++i) {
        // Rounded to float once, here (reference.h): a mean of finite floats
        // 0 or more is within a float's range (ES.46).
        image.rgb[i] = static_cast<float>(mean_[i]);
    }
    return image;
}

namespace {

// The floors' shared precondition.
void check_floor(std::size_t count, const char* which) {
    if (count < 2) {
        throw std::logic_error(std::string{"ReferenceBuilder::"} + which + ": " + std::to_string(count) +
                               " batches, where a floor needs two or more");
    }
}

}  // namespace

double ReferenceBuilder::mse_floor() const {
    check_floor(count_, "mse_floor");
    const double n = static_cast<double>(count_);  // exact (ES.46)
    // Summed in double in pixel order, the same bits every run (GDSA.2).
    double sum = 0.0;
    for (const double m2 : m2_) {
        sum += m2 / (n - 1.0) / n;  // var / n: the mean of n batches' variance
    }
    return sum / static_cast<double>(m2_.size());
}

double ReferenceBuilder::relative_mse_floor() const {
    check_floor(count_, "relative_mse_floor");
    const double n = static_cast<double>(count_);  // exact (ES.46)
    double sum = 0.0;
    for (std::size_t i = 0; i < m2_.size(); ++i) {
        const double scale = mean_[i] + relative_epsilon;  // error.h's epsilon
        sum += m2_[i] / (n - 1.0) / n / (scale * scale);
    }
    return sum / static_cast<double>(m2_.size());
}

}  // namespace serenity::measurement
