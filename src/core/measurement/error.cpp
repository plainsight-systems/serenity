#include "core/measurement/error.h"

#include <cmath>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>

namespace serenity::measurement {

ImageError error_against(Judged judged) {
    const contracts::LinearImage& image = judged.image;
    const contracts::LinearImage& reference = judged.reference;
    // Every refusal before any sum (error.h, I.5): each image's contract,
    // their extents, then each value, in pixel order.
    const std::size_t values = contracts::checked_values(reference, "error_against: the reference");
    (void)contracts::checked_values(image, "error_against: the image");
    if (!(image.extent == reference.extent)) {
        throw std::invalid_argument("error_against: the image is " + std::to_string(image.extent.width) + " x " +
                                    std::to_string(image.extent.height) + ", the reference " +
                                    std::to_string(reference.extent.width) + " x " +
                                    std::to_string(reference.extent.height));
    }
    for (std::size_t i = 0; i < values; ++i) {
        const float v = image.rgb[i];
        const float r = reference.rgb[i];
        if (!std::isfinite(v)) {
            throw std::invalid_argument("error_against: the image's " +
                                        contracts::value_position(image.extent, i) + " is " +
                                        std::format("{}", v) + ", not finite");
        }
        if (!std::isfinite(r) || r < 0.0f) {
            throw std::invalid_argument("error_against: the reference's " +
                                        contracts::value_position(reference.extent, i) + " is " +
                                        std::format("{}", r) + ", where radiance is finite and 0 or more");
        }
    }

    // Summed in double, in pixel order, rows from the top: the same images
    // give the same bits (GDSA.2, run to run). No pixel is skipped (error.h).
    double squared = 0.0;
    double relative = 0.0;
    for (std::size_t i = 0; i < values; ++i) {
        const double difference = double{image.rgb[i]} - double{reference.rgb[i]};
        const double scale = double{reference.rgb[i]} + relative_epsilon;
        squared += difference * difference;
        relative += difference * difference / (scale * scale);
    }
    const auto count = static_cast<double>(values);  // exact: under 2^53 (ES.46)
    return ImageError{.mse = squared / count, .relative_mse = relative / count};
}

}  // namespace serenity::measurement
