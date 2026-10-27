#include "metal/passes/denoise/rungs.h"

#include <cstdint>

#include "core/denoise/unet.h"

namespace serenity::metal {

std::optional<frame::Extent> smallest_rung(std::span<const frame::Extent> rungs,
                                           frame::Extent light) noexcept {
  const frame::Extent grid = denoise::unet_padded(light);
  const auto pixels = [](frame::Extent e) {
    return std::uint64_t{e.width} * e.height;
  };
  std::optional<frame::Extent> best;
  for (const frame::Extent& rung : rungs) {
    if (rung.width < grid.width || rung.height < grid.height) {
      continue;
    }
    if (!best || pixels(rung) < pixels(*best) ||
        (pixels(rung) == pixels(*best) && rung.width < best->width)) {
      best = rung;
    }
  }
  return best;
}

}  // namespace serenity::metal
