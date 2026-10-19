#include "core/presentation/frame_source.h"

#include <cmath>

namespace serenity::presentation {

std::optional<std::string> frame_rate_refusal(FrameRate rate) {
  const bool positive = std::isfinite(rate.minimum) &&
                        std::isfinite(rate.maximum) && rate.minimum > 0.0F;
  if (positive && rate.minimum <= rate.preferred &&
      rate.preferred <= rate.maximum) {
    return std::nullopt;
  }
  return "a frame rate range must be positive and in order (minimum <= "
         "preferred <= maximum)";
}

}  // namespace serenity::presentation
