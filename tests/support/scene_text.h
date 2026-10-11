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
