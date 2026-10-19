// The scene's block (core/scene/scene_block.h, scene_buffers.h): every
// field holds the address of its own array, and a shader reading an array
// through it, at the stride of its own type, reads what the scene reader
// wrote. Checked two ways:
//
//   - word by word, through a probe (kernels/scene_block_probe.metal): every
//     word of every element of every one of the block's arrays, read
//     through the block's field as the shaders' type, against the scene
//     description's bytes. A field given another array's address, two
//     fields' order differing between host and shader, or a type's size
//     differing between them, fails here;
//   - in a frame: the preview over a scene that reads the wood, the swirl,
//     the emissive and the medium arrays, the arrays no other render test
//     reads all of, each shape showing what only its own array gives.

#include "core/scene/scene_block.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "core/camera/thin_lens.h"
#include "core/scene/scene.h"
#include "gpu/kernels/scene_block_probe.h"
#include "gpu/support/probe_runner.h"
#include "gpu/support/rendering.h"
#include "metal/scene/scene_buffers.h"
#include "support/references.h"
#include "support/scene_text.h"
#include "support/vector.h"

namespace gpu = serenity::gpu;
namespace lights = serenity::lights;
namespace materials = serenity::materials;
namespace media = serenity::media;
namespace metal = serenity::metal;
namespace scene = serenity::scene;
namespace shapes = serenity::shapes;
namespace tests = serenity::tests;
namespace textures = serenity::textures;
using tests::Binding;
using tests::BlockQuery;

namespace {

// A scene with at least one element in every one of the block's arrays.
std::string every_kind() {
  return tests::camera_text("[0, 0, 5]", "[0, 0, 0]", 40) +
         tests::sky("[1, 1, 1]", "[1, 1, 1]") + R"(
[textures.checks]
kind = "checker"
size = 0.5
a = [0.9, 0.8, 0.7]
b = [0.1, 0.2, 0.3]
[textures.walnut]
kind = "wood"
light = [0.13, 0.065, 0.03]
dark = [0.045, 0.02, 0.008]
ring = 0.004
board = 0.16
seed = 3
[textures.vanes]
kind = "swirl"
a = [1, 0, 0]
b = [0, 0, 1]
vanes = 2
twist = 0
seed = 4
[materials.floor]
kind = "rough"
texture = "checks"
[materials.table]
kind = "rough"
texture = "walnut"
[materials.core]
kind = "rough"
texture = "vanes"
[materials.glass]
kind = "dielectric"
ior = 1.5
[materials.brass]
kind = "conductor"
f0 = [0.91, 0.78, 0.42]
roughness = 0.35
[materials.glow]
kind = "emissive"
radiance = [0.75, 0.5, 0.25]
[materials.porcelain]
kind = "coated"
color = [0.6, 0.05, 0.04]
ior = 1.5
[media.red_out]
kind = "absorbing"
tint = [0.5, 1, 1]
tint_distance = 1
[[shapes]]
kind = "sphere"
center = [-1, 0, 0]
radius = 0.6
material = "core"
[[shapes]]
kind = "box"
min = [0.4, -0.6, -0.6]
max = [1.6, 0.6, 0.6]
material = "table"
[[shapes]]
kind = "sphere"
center = [0, 1.1, 0]
radius = 0.3
material = "glow"
[[shapes]]
kind = "sphere"
center = [0, -1.1, 0]
radius = 0.35
material = "glass"
interior = "red_out"
[[shapes]]
kind = "sphere"
center = [-3, 3, -6]
radius = 0.3
material = "brass"
[[shapes]]
kind = "sphere"
center = [3, 3, -6]
radius = 0.3
material = "porcelain"
[[shapes]]
kind = "box"
min = [-20, -2.1, -20]
max = [20, -2, 20]
material = "floor"
)";
}

// The 32-bit words of an array, in order.
template <typename T>
std::vector<std::uint32_t> words(std::span<const T> array) {
  static_assert(std::is_trivially_copyable_v<T> && sizeof(T) % 4 == 0,
                "a block array is read as 32-bit words");
  std::vector<std::uint32_t> out(array.size() * sizeof(T) / 4);
  std::memcpy(out.data(), array.data(), out.size() * 4);
  return out;
}

template <typename T>
std::vector<std::uint32_t> words(const std::vector<T>& array) {
  return words(std::span<const T>(array));
}

template <typename T>
std::vector<std::uint32_t> words(const T& one) {
  return words(std::span<const T>(&one, 1));
}

// Each of the block's fields, in its order (core/scene/scene_block.h): a name,
// the size of one element, and the words the scene reader wrote.
struct Field {
  const char* name;
  std::size_t element_size;
  std::vector<std::uint32_t> written;
};

std::vector<Field> fields(const scene::SceneDescription& s) {
  return {
      {"environment", sizeof(s.environment), words(s.environment)},
      {"textures", sizeof(textures::TextureRecord), words(s.textures)},
      {"checkers", sizeof(textures::CheckerData), words(s.checkers)},
      {"woods", sizeof(textures::WoodData), words(s.woods)},
      {"swirls", sizeof(textures::SwirlData), words(s.swirls)},
      {"materials", sizeof(materials::MaterialRecord), words(s.materials)},
      {"rough", sizeof(materials::RoughData), words(s.roughs)},
      {"dielectrics", sizeof(materials::DielectricData), words(s.dielectrics)},
      {"conductors", sizeof(materials::ConductorData), words(s.conductors)},
      {"emissives", sizeof(materials::EmissiveData), words(s.emissives)},
      {"coated", sizeof(materials::CoatedData), words(s.coateds)},
      {"media", sizeof(media::MediumRecord), words(s.media)},
      {"absorbing", sizeof(media::AbsorbingData), words(s.absorbings)},
      {"shapes", sizeof(shapes::ShapeRecord), words(s.shapes.records)},
      {"boxes", sizeof(shapes::BoxData), words(s.shapes.boxes)},
      {"light_records", sizeof(lights::LightRecord), words(s.lights)},
      {"shape_lights", sizeof(std::uint32_t), words(s.shape_lights)},
      {"sphere_lights", sizeof(lights::SphereLightData),
       words(s.sphere_lights)},
      {"light_counts", sizeof(s.light_counts), words(s.light_counts)},
  };
}

}  // namespace

TEST_CASE(
    "every field of the scene's block reads its own array, every word as the "
    "scene reader wrote it") {
  const scene::SceneDescription description =
      scene::parse(every_kind(), "every kind");
  const std::vector<Field> expected = fields(description);
  REQUIRE(expected.size() * sizeof(std::uint64_t) == sizeof(gpu::SceneBlock));

  std::vector<BlockQuery> queries;
  std::vector<std::uint32_t> wanted;
  std::vector<std::size_t> owner;  // the field each query asks of
  for (std::size_t f = 0; f < expected.size(); ++f) {
    INFO("field " << std::string(expected[f].name));
    REQUIRE_FALSE(
        expected[f]
            .written.empty());  // every kind present, so every field is read
    const std::size_t per_element = expected[f].element_size / 4;
    for (std::size_t w = 0; w < expected[f].written.size(); ++w) {
      queries.push_back({static_cast<std::uint32_t>(f),
                         static_cast<std::uint32_t>(w / per_element),
                         static_cast<std::uint32_t>(w % per_element)});
      wanted.push_back(expected[f].written[w]);
      owner.push_back(f);
    }
  }

  tests::ProbeRunner gpu;
  const metal::SceneBuffers buffers(gpu.device(), gpu.submission(),
                                    description);
  const auto count = static_cast<std::uint32_t>(queries.size());
  const std::vector<std::uint32_t> read =
      gpu.run<std::uint32_t>("scene_block_probe", queries.size(),
                             {Binding::of(queries), Binding::of(count),
                              Binding::at(buffers.block_address())});

  std::vector<std::size_t> wrong(expected.size(), 0);
  for (std::size_t i = 0; i < queries.size(); ++i) {
    wrong[owner[i]] += read[i] == wanted[i] ? 0u : 1u;
  }
  for (std::size_t f = 0; f < expected.size(); ++f) {
    INFO("field " << std::string(expected[f].name) << ": "
                  << expected[f].written.size() << " words");
    CHECK(wrong[f] == 0);
  }
}
