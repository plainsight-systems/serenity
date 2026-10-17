#pragma once

// Whether two images, or two runs of floats, are the same to the bit (ES.3):
// their bytes compared, so -0 differs from +0 and a NaN is the same as
// itself, which is what "to the bit" means, where == on floats says
// neither.

#include <cstring>
#include <span>

#include "core/contracts/linear_image.h"

namespace serenity::tests {

inline bool same_bits(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

inline bool same_bits(const contracts::LinearImage& a, const contracts::LinearImage& b) {
    return a.extent == b.extent && same_bits(a.rgb, b.rgb);
}

}  // namespace serenity::tests
