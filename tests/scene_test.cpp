// Scene files: what they read into, and every mistake refused by file and
// line.

#include <string>

#include <doctest/doctest.h>

#include "core/scene/scene.h"

using namespace serenity;

namespace {

// The example in core/scene/scene.h.
constexpr const char* example = R"(
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

std::string error_of(const std::string& text) {
    try {
        (void)scene::parse(text, "s.toml");
    } catch (const scene::Error& error) {
        return error.what();
    }
    return "";
}

// `example` with the first occurrence of `from` replaced by `to`.
std::string with(const std::string& from, const std::string& to) {
    std::string text = example;
    const std::size_t at = text.find(from);
    REQUIRE(at != std::string::npos);
    return text.replace(at, from.size(), to);
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

TEST_CASE("the example reads into one array per kind, names resolved to indices") {
    const scene::SceneDescription s = scene::parse(example, "example");

    CHECK(s.camera.position.y == doctest::Approx(1.2f));
    CHECK(s.camera.vertical_fov_degrees == 40.0f);
    CHECK(s.environment.zenith.z == doctest::Approx(0.08f));

    REQUIRE(s.textures.size() == 1);
    CHECK(s.textures[0].kind == textures::TextureKind::checker);
    REQUIRE(s.checkers.size() == 1);
    CHECK(s.checkers[0].size == 0.5f);

    // Materials in name order: floor, then glass.
    REQUIRE(s.materials.size() == 2);
    CHECK(s.materials[0].kind == materials::MaterialKind::rough);
    CHECK(s.materials[1].kind == materials::MaterialKind::dielectric);
    REQUIRE(s.rough.size() == 1);
    CHECK(s.rough[0].texture.index == 0);
    REQUIRE(s.dielectrics.size() == 1);
    CHECK(s.dielectrics[0].ior == 1.5f);

    // Shapes in file order, each naming its material by index.
    REQUIRE(s.shapes.records.size() == 2);
    CHECK(s.shapes.records[0].kind == shapes::ShapeKind::sphere);
    CHECK(s.shapes.records[1].kind == shapes::ShapeKind::box);
    CHECK(s.shapes.spheres[0].material == 1);
    CHECK(s.shapes.boxes[0].material == 0);
    CHECK(s.shapes.boxes[0].min.x == -6.0f);
}

TEST_CASE("up defaults to +y, and a rough material may have a color instead of a texture") {
    const scene::SceneDescription s =
        scene::parse(with("texture = \"floor_checks\"", "color = [0.5, 0.25, 1]"), "s");
    CHECK(s.rough[0].texture.index == contracts::no_texture);
    CHECK(s.rough[0].color.y == 0.25f);

    const scene::SceneDescription no_up = scene::parse(with("up = [0, 1, 0]\n", ""), "s");
    CHECK(no_up.camera.up.y == 1.0f);
    CHECK(no_up.camera.up.x == 0.0f);
}

TEST_CASE("the first scene reads, with its fireflies as lights") {
    const scene::SceneDescription s = scene::load(SERENITY_SCENES_DIR "/brass_sphere.toml");
    CHECK(s.shapes.records.size() == 4);
    CHECK(s.conductors.size() == 1);
    CHECK(s.sphere_lights.size() == 2);
    CHECK(s.light_counts.spheres == 2);
}

TEST_CASE("metal, glow, and every glowing sphere a light") {
    const std::string text = with("[[shapes]]", R"([materials.brass]
kind = "conductor"
f0 = [0.91, 0.78, 0.42]
roughness = 0.35

[materials.glow]
kind = "emissive"
radiance = [40, 36, 12]

[[shapes]]
kind = "sphere"
center = [1, 2, 3]
radius = 0.05
material = "glow"

[[shapes]]
kind = "sphere"
center = [2, 0.5, 0]
radius = 0.5
material = "brass"

[[shapes]])");
    const scene::SceneDescription s = scene::parse(text, "s");
    // Materials in name order: brass, floor, glass, glow.
    REQUIRE(s.materials.size() == 4);
    CHECK(s.materials[0].kind == materials::MaterialKind::conductor);
    CHECK(s.materials[3].kind == materials::MaterialKind::emissive);
    REQUIRE(s.conductors.size() == 1);
    CHECK(s.conductors[0].f0.y == doctest::Approx(0.78f));
    CHECK(s.conductors[0].roughness == doctest::Approx(0.35f));
    REQUIRE(s.emissives.size() == 1);
    CHECK(s.emissives[0].radiance.x == 40.0f);

    // The glowing sphere is the first shape, and the one light.
    REQUIRE(s.sphere_lights.size() == 1);
    CHECK(s.light_counts.spheres == 1);
    CHECK(s.sphere_lights[0].primitive == 0);
    CHECK(s.sphere_lights[0].center.z == 3.0f);
    CHECK(s.sphere_lights[0].radius == doctest::Approx(0.05f));
    CHECK(s.sphere_lights[0].radiance.y == 36.0f);
}

TEST_CASE("every mistake is refused, naming the file and the line") {
    CHECK(contains(error_of(with("radius = 0.8", "radius = 0")), "s.toml:30: shape 1's radius must be greater than 0"));
    CHECK(contains(error_of(with("ior = 1.5", "ior = 1")), "ior must be greater than 1"));
    CHECK(contains(error_of(with("size = 0.5", "size = -1")), "size must be greater than 0"));
    CHECK(contains(error_of(with("max = [6, 0, 6]", "max = [6, -0.1, 6]")), "min must be below its max"));
    CHECK(contains(error_of(with("look_at = [0, 0.8, 0]", "look_at = [0, 1.2, 4]")), "cannot be framed"));
    CHECK(contains(error_of(with("vertical_fov_degrees = 40", "vertical_fov_degrees = 180")), "cannot be framed"));
    CHECK(contains(error_of(with("kind = \"sphere\"", "kind = \"cone\"")), "unknown shape kind 'cone'"));
    CHECK(contains(error_of(with("kind = \"dielectric\"", "kind = \"metal\"")), "unknown material kind 'metal'"));
    CHECK(contains(error_of(with("kind = \"checker\"", "kind = \"wood\"")), "unknown texture kind 'wood'"));
    CHECK(contains(error_of(with("kind = \"gradient\"", "kind = \"stars\"")), "unknown environment kind 'stars'"));
    CHECK(contains(error_of(with("material = \"glass\"", "material = \"crystal\"")),
                   "uses material 'crystal', which is not defined"));
    CHECK(contains(error_of(with("texture = \"floor_checks\"", "texture = \"tiles\"")),
                   "uses texture 'tiles', which is not defined"));
    CHECK(contains(error_of(with("texture = \"floor_checks\"", "texture = \"floor_checks\"\ncolor = [1, 1, 1]")),
                   "exactly one of 'color' and 'texture'"));
    CHECK(contains(error_of(with("radius = 0.8", "radius = 0.8\nshiny = true")), "unknown key 'shiny' in shape 1"));
    CHECK(contains(error_of(with("[camera]", "[camera]\nzoom = 2")), "unknown key 'zoom' in [camera]"));
    CHECK(contains(error_of(with("[environment]", "[lights]\n[environment]")), "unknown key 'lights' in the scene"));
    CHECK(contains(error_of(with("radius = 0.8", "radius = \"big\"")), "must be a number"));
    // Finite as a double, beyond float's range: refused, not narrowed to infinity.
    CHECK(contains(error_of(with("radius = 0.8", "radius = 1e300")), "within float's range"));
    CHECK(contains(error_of(with("ior = 1.5", "ior = 1e39")), "within float's range"));
    CHECK(contains(error_of(with("center = [0, 0.8, 0]", "center = [0, -1e300, 0]")), "within float's range"));
    CHECK(contains(error_of(with("center = [0, 0.8, 0]", "center = [0, 0.8]")), "array of three numbers"));
    CHECK(contains(error_of(with("radius = 0.8\n", "")), "shape 1 has no 'radius'"));
    CHECK(contains(error_of("[camera]\nposition = [0, 0, 1]\nlook_at = [0, 0, 0]\nvertical_fov_degrees = 40\n"
                            "[environment]\nkind = \"gradient\"\nzenith = [0, 0, 0]\nhorizon = [0, 0, 0]\n"),
                   "no 'shapes'"));
    CHECK(contains(error_of("[camera\n"), "s.toml:1:"));

    const std::string brass = "[materials.brass]\nkind = \"conductor\"\n";
    CHECK(contains(error_of(with("[[shapes]]", brass + "f0 = [1.2, 0.8, 0.4]\nroughness = 0.3\n[[shapes]]")),
                   "f0 must be within [0, 1]"));
    CHECK(contains(error_of(with("[[shapes]]", brass + "f0 = [0.9, 0.8, 0.4]\nroughness = 0\n[[shapes]]")),
                   "roughness must be in (0, 1]"));
    const std::string glow = "[materials.glow]\nkind = \"emissive\"\n";
    CHECK(contains(error_of(with("[[shapes]]", glow + "radiance = [1, -1, 1]\n[[shapes]]")),
                   "radiance must not be negative"));
    std::string glowing_box = with("[[shapes]]", glow + "radiance = [1, 1, 1]\n[[shapes]]");
    const std::size_t floor = glowing_box.rfind("material = \"floor\"");
    glowing_box.replace(floor, std::string("material = \"floor\"").size(), "material = \"glow\"");
    CHECK(contains(error_of(glowing_box), "only a sphere may be emissive"));
}

TEST_CASE("a missing file is refused by name") {
    CHECK_THROWS_WITH_AS(scene::load("no/such/scene.toml"), doctest::Contains("no/such/scene.toml"), scene::Error);
}
