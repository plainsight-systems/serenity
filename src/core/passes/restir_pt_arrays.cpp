#include "core/passes/restir_pt_arrays.h"

#include "core/contracts/float3.h"
#include "core/contracts/light_tile.h"
#include "core/contracts/path_reservoir.h"
#include "core/contracts/path_reuse_settings.h"
#include "core/contracts/primary_hit.h"

namespace serenity::passes {

PathReuseArrays path_reuse_arrays(frame::Extent size, bool spatial,
                                  bool temporal) {
  const std::size_t pixels = std::size_t{size.width} * size.height;
  PathReuseArrays arrays;
  arrays.copies = temporal ? 2 : 1;
  arrays.reservoirs = pixels * sizeof(contracts::PathReservoir);
  arrays.primaries = pixels * sizeof(contracts::PrimaryHit);
  arrays.light = pixels * 4 * sizeof(float);
  if (spatial || temporal) {
    arrays.shifts = pixels * contracts::max_path_neighbors * 4 * sizeof(float);
    arrays.replays =
        pixels * contracts::max_path_neighbors * sizeof(std::uint32_t);
  }
  if (temporal) {
    arrays.finals = pixels * sizeof(contracts::PathReservoir);
    arrays.previous_positions = pixels * sizeof(contracts::Float3);
    arrays.previous_pixels = pixels * sizeof(std::uint32_t);
    arrays.duplication = pixels * sizeof(float);
  }
  return arrays;
}

std::size_t light_tiles_bytes() {
  return std::size_t{contracts::light_tiles} * contracts::light_tile_size *
         sizeof(contracts::LightTileEntry);
}

std::size_t light_power_bytes(std::uint32_t lights) {
  return contracts::light_power_floats(contracts::light_power_levels(lights)) *
         sizeof(float);
}

}  // namespace serenity::passes
