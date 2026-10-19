#pragma once

// Scene files written in the tests (core/scene/scene.h), in parts (ES.3,
// F.10): a number written exactly, a camera, a sky, and the scenes more than
// one test file reads or renders; the tests' own, so none leans on what
// scenes/ holds.

#include <doctest/doctest.h>

#include <array>
#include <charconv>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>

namespace serenity::tests {

// `v` written so that reading it back gives `v` exactly: the shortest such
// text (std::to_chars), for a float as for a double.
template <typename Real>
std::string exact(Real v) {
  std::array<char, 32> text{};
  const std::to_chars_result written =
      std::to_chars(text.data(), text.data() + text.size(), v);
  REQUIRE(written.ec == std::errc{});
  return {text.data(), written.ptr};
}

inline std::string camera_text(std::string_view position,
                               std::string_view look_at, int fov_degrees) {
  return "[camera]\nposition = " + std::string(position) +
         "\nlook_at = " + std::string(look_at) +
         "\nvertical_fov_degrees = " + std::to_string(fov_degrees) + "\n";
}

// A gradient sky; zenith and horizon alike, a sky of one radiance every way.
inline std::string sky(std::string_view zenith, std::string_view horizon) {
  return "[environment]\nkind = \"gradient\"\nzenith = " + std::string(zenith) +
         "\nhorizon = " + std::string(horizon) + "\n";
}

// The example in core/scene/scene.h: a glass sphere on a floor of checkers.
// Its lines, for the errors that name them: the sphere's are 28 to 31, the
// box's 34 to 37.
inline constexpr const char* scene_example = R"(
[camera]
position = [0, 1.2, 4]
look_at = [0, 0.8, 0]
up = [0, 1, 0]
vertical_fov_degrees = 40

[environment]
kind = "gradient"
zenith = [0.02, 0.03, 0.08]
horizon = [0.15, 0.17, 0.25]

[textures.floor_checks]
kind = "checker"
size = 0.5
a = [0.9, 0.9, 0.9]
b = [0.1, 0.1, 0.1]

[materials.glass]
kind = "dielectric"
ior = 1.5

[materials.floor]
kind = "rough"
texture = "floor_checks"

[[shapes]]
kind = "sphere"
center = [0, 0.8, 0]
radius = 0.8
material = "glass"

[[shapes]]
kind = "box"
min = [-6, -0.1, -6]
max = [6, 0, 6]
material = "floor"
)";

// A sphere of radius 1 at the origin of `material` (its kind and values,
// one per line), under a sky of 0.5 every way, seen from 4 away: the
// furnace, which a sphere that returns all it receives vanishes into.
inline std::string furnace(std::string_view material) {
  return camera_text("[0, 0, 4]", "[0, 0, 0]", 30) +
         sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") + "[materials.m]\n" +
         std::string(material) +
         "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = "
         "1\nmaterial = \"m\"\n";
}

// A brass ball on a checkered floor under a dusk sky, and fireflies about
// it: the tests' own scene of moving, glowing lights (core/animation/), so
// no test leans on what scenes/ holds. `fireflies`' shapes follow the ball's
// and the floor's (shapes 0 and 1).
inline std::string brass_ball_under(std::string_view fireflies) {
  return camera_text("[0.0, 1.1, 4.2]", "[0.0, 0.75, 0.0]", 38) +
         sky("[0.01, 0.015, 0.06]", "[0.06, 0.08, 0.175]") +
         "[textures.floor_checks]\nkind = \"checker\"\nsize = 0.4\na = [0.75, "
         "0.72, 0.65]\nb = [0.05, 0.05, 0.06]\n"
         "[materials.brass]\nkind = \"conductor\"\nf0 = [0.91, 0.78, "
         "0.42]\nroughness = 0.35\n"
         "[materials.floor]\nkind = \"rough\"\ntexture = \"floor_checks\"\n"
         "[materials.firefly]\nkind = \"emissive\"\nradiance = [320.0, 280.0, "
         "70.0]\n"
         "[[shapes]]\nkind = \"sphere\"\nname = \"brass\"\ncenter = [0.0, "
         "0.75, 0.0]\nradius = 0.75\n"
         "material = \"brass\"\n"
         "[[shapes]]\nkind = \"box\"\nmin = [-200.0, -0.1, -200.0]\nmax = "
         "[200.0, 0.0, 200.0]\nmaterial = \"floor\"\n" +
         std::string(fireflies);
}

// Three fireflies flying about the ball and flashing as they fly: every
// frame, shapes move (the structure rebuilt) and lights glow.
inline std::string flying_fireflies() {
  // Each one's center, then its flight's behaviours and seed.
  const std::array<std::array<std::string_view, 2>, 3> each = {
      std::array<std::string_view, 2>{
          "[-1.05, 0.55, 0.85]", "circle = 3, swoop = 1, drift = 1, seed = 1"},
      std::array<std::string_view, 2>{
          "[0.95, 1.45, -0.55]", "circle = 3, swoop = 1, drift = 1, seed = 2"},
      std::array<std::string_view, 2>{
          "[-1.6, 1.2, -1.2]", "circle = 1, swoop = 2, drift = 1, seed = 3"}};
  std::string fireflies;
  for (const auto& [center, behaviours] : each) {
    fireflies +=
        "[[shapes]]\nkind = \"sphere\"\ncenter = " + std::string(center) +
        "\nradius = 0.03\nmaterial = \"firefly\"\n"
        "motion = { kind = \"flight\", min = [-2.2, 0.12, -2.2], max = [2.2, "
        "2.2, 2.2], "
        "targets = [\"brass\"], speed = 0.45, clearance = 0.06, " +
        std::string(behaviours) +
        " }\nglow = { kind = \"flight\", flash = 0.35, dim = 0.25 }\n";
  }
  return brass_ball_under(fireflies);
}

// Two fireflies wandering about, steady: shapes that move, no glow.
inline std::string wandering_fireflies() {
  return brass_ball_under(
      "[[shapes]]\nkind = \"sphere\"\ncenter = [-1.05, 0.55, 0.85]\nradius = "
      "0.035\nmaterial = \"firefly\"\n"
      "motion = { kind = \"wander\", reach = 0.25, speed = 0.25, seed = 1 }\n"
      "[[shapes]]\nkind = \"sphere\"\ncenter = [0.95, 1.45, -0.55]\nradius = "
      "0.03\nmaterial = \"firefly\"\n"
      "motion = { kind = \"wander\", reach = 0.25, speed = 0.3, seed = 2 }\n");
}

// Two fireflies held still: nothing moves or glows.
inline std::string still_fireflies() {
  return brass_ball_under(
      "[[shapes]]\nkind = \"sphere\"\ncenter = [-1.05, 0.55, 0.85]\nradius = "
      "0.035\nmaterial = \"firefly\"\n"
      "[[shapes]]\nkind = \"sphere\"\ncenter = [0.95, 1.45, -0.55]\nradius = "
      "0.03\nmaterial = \"firefly\"\n");
}

}  // namespace serenity::tests
