#include "core/scene/packed_range.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>

#include "core/contracts/bsdf.h"

namespace serenity::scene {

namespace {

// Whether `value` is within the packed form's precondition (packed_range.h).
bool fits(float value) {
  return std::isfinite(value) && std::abs(value) <= contracts::packed_largest;
}

bool fits(contracts::Float3 c) { return fits(c.x) && fits(c.y) && fits(c.z); }

// The largest a color's channels reach, in magnitude.
float largest(contracts::Float3 c) {
  return std::max({std::abs(c.x), std::abs(c.y), std::abs(c.z)});
}

// A float with the digits that read back as itself, so a value just past
// 65504 is not printed as 65504.
std::string number(float value) {
  std::ostringstream out;
  out << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return out.str();
}

std::string triple(contracts::Float3 c) {
  return "(" + number(c.x) + ", " + number(c.y) + ", " + number(c.z) + ")";
}

std::string material_name(const SceneDescription& scene, std::size_t i) {
  return i < scene.material_names.size()
             ? "material '" + scene.material_names[i] + "'"
             : "material " + std::to_string(i);
}

std::string outside(const std::string& what, const std::string& value) {
  return what + ", " + value +
         ", is outside what ReSTIR PT's packed G-buffer holds: finite, at "
         "most " +
         number(contracts::packed_largest) + " in magnitude (contracts/bsdf.h)";
}

// A base's color, or its texture's colors as the texture can return them
// (packed_range.h).
std::optional<std::string> base_outside(const SceneDescription& scene,
                                        const std::string& what,
                                        contracts::Float3 color,
                                        contracts::TextureReference texture) {
  if (texture.index == contracts::no_texture) {
    return fits(color)
               ? std::nullopt
               : std::optional{outside(what + "'s color", triple(color))};
  }
  const textures::TextureRecord record = scene.textures.at(texture.index);
  contracts::Float3 a{};
  contracts::Float3 b{};
  float scale = 1.0f;
  switch (record.kind) {
    case textures::TextureKind::checker:
      a = scene.checkers.at(record.index).a;
      b = scene.checkers.at(record.index).b;
      break;
    case textures::TextureKind::wood:
      a = scene.woods.at(record.index).light;
      b = scene.woods.at(record.index).dark;
      scale = 1.0f + textures::wood_board_shade;
      break;
    case textures::TextureKind::swirl:
      a = scene.swirls.at(record.index).a;
      b = scene.swirls.at(record.index).b;
      break;
  }
  if (!fits(a) || !fits(b) || !fits(scale * std::max(largest(a), largest(b)))) {
    return outside(what + "'s texture's colors",
                   triple(a) + " and " + triple(b) +
                       (scale == 1.0f ? "" : " times up to " + number(scale)));
  }
  return std::nullopt;
}

std::optional<std::string> scalar_outside(const std::string& what,
                                          std::string_view name, float value) {
  return fits(value) ? std::nullopt
                     : std::optional{outside(what + "'s " + std::string(name),
                                             number(value))};
}

}  // namespace

std::optional<std::string> outside_packed_bsdf(const SceneDescription& scene) {
  for (std::size_t i = 0; i < scene.materials.size(); ++i) {
    const materials::MaterialRecord record = scene.materials[i];
    const std::string what = material_name(scene, i);
    std::optional<std::string> found;
    switch (record.kind) {
      case materials::MaterialKind::rough: {
        const materials::RoughData& rough = scene.roughs.at(record.index);
        found = base_outside(scene, what, rough.color, rough.texture);
        break;
      }
      case materials::MaterialKind::coated: {
        const materials::CoatedData& coated = scene.coateds.at(record.index);
        found = base_outside(scene, what, coated.color, coated.texture);
        if (!found) {
          found = scalar_outside(what, "ior", coated.ior);
        }
        if (!found) {
          found = scalar_outside(what, "escape", coated.escape);
        }
        break;
      }
      case materials::MaterialKind::conductor: {
        const contracts::Float3 f0 = scene.conductors.at(record.index).f0;
        found = fits(f0) ? std::nullopt
                         : std::optional{outside(what + "'s f0", triple(f0))};
        break;
      }
      case materials::MaterialKind::dielectric:
        found =
            scalar_outside(what, "ior", scene.dielectrics.at(record.index).ior);
        break;
      case materials::MaterialKind::emissive:
        break;
    }
    if (found) {
      return found;
    }
  }
  return std::nullopt;
}

}  // namespace serenity::scene
