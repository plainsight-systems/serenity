#pragma once

// Scene files written in the tests (core/scene/scene.h), in parts (ES.3,
// F.10): a number written exactly, a camera, a sky, and the scenes more than
// one test file reads or renders.

#include <array>
#include <charconv>
#include <string>
#include <string_view>
#include <system_error>

#include <doctest/doctest.h>

namespace serenity::tests {

// `v` written so that reading it back gives `v` exactly: the shortest such
// text (std::to_chars), for a float as for a double.
template <typename Real>
std::string exact(Real v) {
    std::array<char, 32> text{};
    const std::to_chars_result written = std::to_chars(text.data(), text.data() + text.size(), v);
    REQUIRE(written.ec == std::errc{});
    return {text.data(), written.ptr};
}

inline std::string camera_text(std::string_view position, std::string_view look_at, int fov_degrees) {
    return "[camera]\nposition = " + std::string(position) + "\nlook_at = " + std::string(look_at) +
           "\nvertical_fov_degrees = " + std::to_string(fov_degrees) + "\n";
}

// A gradient sky; zenith and horizon alike, a sky of one radiance every way.
inline std::string sky(std::string_view zenith, std::string_view horizon) {
    return "[environment]\nkind = \"gradient\"\nzenith = " + std::string(zenith) + "\nhorizon = " +
           std::string(horizon) + "\n";
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
    return camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
           "[materials.m]\n" + std::string(material) +
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"m\"\n";
}

// A light at the most radiance a float holds, inside a closed white sphere,
// the camera between them: a path tracer's light overflows a float. The
// cavity loses light only to the light, so its wall's radiance is the
// light's own, and a path's sum over its surfaces passes the largest float
// in about a third of the samples. A sum, not one term: the path tracer's
// MIS bounds a light sample's term at a Lambert surface by half the light's
// radiance times the albedo (integrator/path.metal.h), so lights merely
// close over a floor, under a black sky, no longer overflow it.
inline std::string blinding_lights() {
    return camera_text("[0, 0, 0.8]", "[0, 0, 1]", 60) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.wall]\nkind = \"rough\"\ncolor = [1, 1, 1]\n"
           "[materials.blinding]\nkind = \"emissive\"\nradiance = [3e38, 3e38, 3e38]\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"wall\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.5\nmaterial = \"blinding\"\n";
}

// Glass of radius 1 filled with a medium keeping half its red over 1 m;
// inside it an opaque white core of radius 0.3, and a light of radius 0.05
// at (0, 0.45, 0.45), which nothing hides from the core's front. The sky is
// black, so the middle of the image is the core, lit by that light, seen
// through the glass. Red is dimmed over the camera's stretch to the core,
// 0.7, and over the light's to the core, |(0, 0.45, 0.15)| - 0.05 = 0.424;
// green over neither.
inline std::string core_lit_inside_glass() {
    return camera_text("[0, 0, 4]", "[0, 0, 0]", 30) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
           "[materials.white]\nkind = \"rough\"\ncolor = [0.8, 0.8, 0.8]\n"
           "[materials.glow]\nkind = \"emissive\"\nradiance = [40, 40, 40]\n"
           "[media.red_out]\nkind = \"absorbing\"\ntint = [0.5, 1, 1]\ntint_distance = 1\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 1\nmaterial = \"glass\"\n"
           "interior = \"red_out\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.3\nmaterial = \"white\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.45, 0.45]\nradius = 0.05\nmaterial = \"glow\"\n";
}

// The stretches core_lit_inside_glass()'s red is dimmed over, in meters:
// the camera's to the core, and the light's.
inline constexpr double camera_to_core = 0.7;
inline constexpr double light_to_core = 0.424;

// A bead of radius 1 mm, filled with a medium keeping exp(-0.7) of its red
// over 1 mm: 700 per meter, seen along z, 5 m from the origin. There a ray
// starts 256 ulps of 5 off a surface it leaves along z, 0.12 mm
// (integrator/surface.metal.h), 8% of a crossing's light; the stretch must
// be measured from the surface itself. (At the origin the step is 15 um,
// 1%, too little to tell.)
inline std::string tinted_bead() {
    return camera_text("[0, 0, 5.004]", "[0, 0, 5]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
           "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
           "[media.red_out]\nkind = \"absorbing\"\ntint = [0.4965853, 1, 1]\ntint_distance = 0.001\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 5]\nradius = 0.001\nmaterial = \"glass\"\n"
           "interior = \"red_out\"\n";
}

}  // namespace serenity::tests
