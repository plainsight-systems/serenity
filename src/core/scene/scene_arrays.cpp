#include "core/scene/scene_arrays.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace serenity::scene {

namespace {

using Block = gpu::SceneBlock;
using Scene = SceneDescription;

// The scene's layouts are copied as bytes, so each must be one whose bytes
// are its value (SL.con.4, COPY.6).
template <typename T>
std::span<const std::byte> bytes(const std::vector<T>& array) {
  static_assert(std::is_trivially_copyable_v<T>,
                "a scene array is copied to the GPU as its bytes");
  return std::as_bytes(std::span(array));
}

template <typename T>
std::span<const std::byte> bytes(const T& one) {
  static_assert(std::is_trivially_copyable_v<T>,
                "a scene record is copied to the GPU as its bytes");
  return std::as_bytes(std::span(&one, 1));
}

// The list (scene_arrays.h): each array with the field of the scene's
// block that holds its address, named, so no array can be given another's
// address. Checked at compile time rather than at run time (P.5), below:
// there is an entry for every field of the block, since the list's length
// is the count of entries written and the block holds that many
// addresses; and no entry is empty or names a field another names.
struct Entry {
  std::uint64_t Block::* field;
  std::span<const std::byte> (*bytes)(const Scene&);
};

constexpr auto entries = std::to_array<Entry>({
    {&Block::environment, [](const Scene& s) { return bytes(s.environment); }},
    {&Block::textures, [](const Scene& s) { return bytes(s.textures); }},
    {&Block::checkers, [](const Scene& s) { return bytes(s.checkers); }},
    {&Block::woods, [](const Scene& s) { return bytes(s.woods); }},
    {&Block::swirls, [](const Scene& s) { return bytes(s.swirls); }},
    {&Block::materials, [](const Scene& s) { return bytes(s.materials); }},
    {&Block::roughs, [](const Scene& s) { return bytes(s.roughs); }},
    {&Block::dielectrics, [](const Scene& s) { return bytes(s.dielectrics); }},
    {&Block::conductors, [](const Scene& s) { return bytes(s.conductors); }},
    {&Block::emissives, [](const Scene& s) { return bytes(s.emissives); }},
    {&Block::coateds, [](const Scene& s) { return bytes(s.coateds); }},
    {&Block::media, [](const Scene& s) { return bytes(s.media); }},
    {&Block::absorbings, [](const Scene& s) { return bytes(s.absorbings); }},
    {&Block::shapes, [](const Scene& s) { return bytes(s.shapes.records); }},
    {&Block::boxes, [](const Scene& s) { return bytes(s.shapes.boxes); }},
    {&Block::light_records, [](const Scene& s) { return bytes(s.lights); }},
    {&Block::shape_lights,
     [](const Scene& s) { return bytes(s.shape_lights); }},
    {&Block::sphere_lights,
     [](const Scene& s) { return bytes(s.sphere_lights); }},
    {&Block::light_counts,
     [](const Scene& s) { return bytes(s.light_counts); }},
});

constexpr bool every_entry_its_own() {
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (entries.at(i).field == nullptr || entries.at(i).bytes == nullptr) {
      return false;
    }
    for (std::size_t j = i + 1; j < entries.size(); ++j) {
      if (entries.at(i).field == entries.at(j).field) {
        return false;
      }
    }
  }
  return true;
}

static_assert(entries.size() == scene_array_count,
              "every field of gpu::SceneBlock needs its entry in `entries`");
static_assert(every_entry_its_own(),
              "an entry of `entries` is empty, or names a field another names");

}  // namespace

std::array<std::span<const std::byte>, scene_array_count> scene_arrays(
    const SceneDescription& scene) {
  std::array<std::span<const std::byte>, scene_array_count> arrays;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    arrays.at(i) = entries.at(i).bytes(scene);
  }
  return arrays;
}

gpu::SceneBlock scene_block(
    const std::array<std::uint64_t, scene_array_count>& addresses) {
  gpu::SceneBlock block{};
  for (std::size_t i = 0; i < entries.size(); ++i) {
    block.*entries.at(i).field = addresses.at(i);
  }
  return block;
}

}  // namespace serenity::scene
