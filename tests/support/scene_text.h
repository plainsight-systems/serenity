#pragma once

// Scene files written in the tests (core/scene/scene.h), in parts (ES.3,
// F.10): a camera, a sky, and the scenes more than one test file renders.

#include <string>

namespace serenity::tests {

inline std::string camera_text(const char* position, const char* look_at, int fov_degrees) {
    return std::string("[camera]\nposition = ") + position + "\nlook_at = " + look_at +
           "\nvertical_fov_degrees = " + std::to_string(fov_degrees) + "\n";
}

// A gradient sky; zenith and horizon alike, a sky of one radiance every way.
inline std::string sky(const char* zenith, const char* horizon) {
    return std::string("[environment]\nkind = \"gradient\"\nzenith = ") + zenith + "\nhorizon = " + horizon + "\n";
}

// Two lights at the most radiance a float holds, close over a floor: a path
// tracer's light from them overflows a float.
inline std::string blinding_lights() {
    return camera_text("[0, 1, 2]", "[0, 0, 0]", 60) + sky("[0, 0, 0]", "[0, 0, 0]") +
           "[materials.floor]\nkind = \"rough\"\ncolor = [0.9, 0.9, 0.9]\n"
           "[materials.blinding]\nkind = \"emissive\"\nradiance = [3e38, 3e38, 3e38]\n"
           "[[shapes]]\nkind = \"box\"\nmin = [-5, -1, -5]\nmax = [5, 0, 5]\nmaterial = \"floor\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0.3, 0]\nradius = 0.25\nmaterial = \"blinding\"\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0.6, 0.3, 0]\nradius = 0.25\nmaterial = \"blinding\"\n";
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
// over 1 mm: 700 per meter. At this scale the 0.1 mm a ray starts off a
// surface is 7% of a crossing's light; the stretch must be measured from the
// surface itself.
inline std::string tinted_bead() {
    return camera_text("[0, 0, 0.004]", "[0, 0, 0]", 30) + sky("[0.5, 0.5, 0.5]", "[0.5, 0.5, 0.5]") +
           "[materials.glass]\nkind = \"dielectric\"\nior = 1.5\n"
           "[media.red_out]\nkind = \"absorbing\"\ntint = [0.4965853, 1, 1]\ntint_distance = 0.001\n"
           "[[shapes]]\nkind = \"sphere\"\ncenter = [0, 0, 0]\nradius = 0.001\nmaterial = \"glass\"\n"
           "interior = \"red_out\"\n";
}

}  // namespace serenity::tests
