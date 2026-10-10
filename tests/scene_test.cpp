// Scene files: what they read into, and every mistake refused by file and
// line.

#include <cmath>
#include <cstddef>
#include <string>
#include <string_view>

#include <doctest/doctest.h>

#include "core/contracts/medium.h"
#include "core/materials/coated.h"
#include "core/scene/scene.h"
#include "support/text.h"
#include "test_paths.h"

using namespace serenity;
using tests::contains;
using tests::replaced;

namespace {

// The example in core/scene/scene.h. Its lines, for the errors that name
// them: the sphere's are 28 to 31, the box's 34 to 37.
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

// The error scene::parse() throws for `text`, read as s.toml, or "" if it
// throws none.
std::string error_of(const std::string& text) {
    return tests::error_of<scene::Error>([&] { return scene::parse(text, "s.toml"); });
}

// `example` with its first `from` replaced by `to`.
std::string with(std::string_view from, std::string_view to) {
    return replaced(example, from, to);
}

}  // namespace

TEST_CASE("the example reads into one array per kind, names resolved to indices") {
    const scene::SceneDescription s = scene::parse(example, "example");

    CHECK(s.camera.position.y == doctest::Approx(1.2f).scale(0).epsilon(1e-6));
    CHECK(s.camera.vertical_fov_degrees == 40.0f);
    CHECK(s.environment.zenith.z == doctest::Approx(0.08f).scale(0).epsilon(1e-6));

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
    CHECK(s.shapes.records[0].material == 1);
    CHECK(s.shapes.records[1].material == 0);

    // Each an instance of a geometry, placed by its transform: the sphere
    // the unit sphere, scaled by its radius and moved to its center; the
    // box its corners as the file gives them, placed by the identity.
    REQUIRE(s.shapes.transforms.size() == 2);
    CHECK(s.shapes.transforms[0].m[0][0] == doctest::Approx(0.8f).scale(0).epsilon(1e-6));
    CHECK(s.shapes.transforms[0].m[2][2] == doctest::Approx(0.8f).scale(0).epsilon(1e-6));
    CHECK(s.shapes.transforms[0].m[1][3] == doctest::Approx(0.8f).scale(0).epsilon(1e-6));
    CHECK(s.shapes.records[1].geometry == 0);
    REQUIRE(s.shapes.boxes.size() == 1);
    CHECK(s.shapes.boxes[0].min.x == -6.0f);
    CHECK(s.shapes.boxes[0].min.y == -0.1f);
    CHECK(s.shapes.boxes[0].max.z == 6.0f);
    CHECK(s.shapes.transforms[1].m[0][0] == 1.0f);
    CHECK(s.shapes.transforms[1].m[0][3] == 0.0f);
    CHECK(s.shapes.transforms[1].m[1][3] == 0.0f);

    // Nothing moves.
    CHECK_FALSE(animation::moves(s.animation));
}

TEST_CASE("up defaults to +y, and a rough material may have a color instead of a texture") {
    const scene::SceneDescription s = scene::parse(with("texture = \"floor_checks\"", "color = [0.5, 0.25, 1]"), "s");
    CHECK(s.rough[0].texture.index == contracts::no_texture);
    CHECK(s.rough[0].color.y == 0.25f);

    const scene::SceneDescription no_up = scene::parse(with("up = [0, 1, 0]\n", ""), "s");
    CHECK(no_up.camera.up.y == 1.0f);
    CHECK(no_up.camera.up.x == 0.0f);
}

TEST_CASE("the first scene reads, with its fireflies as lights") {
    const scene::SceneDescription s = scene::load(tests::scenes_dir / "brass_sphere.toml");
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
    CHECK(s.conductors[0].f0.y == doctest::Approx(0.78f).scale(0).epsilon(1e-6));
    CHECK(s.conductors[0].roughness == doctest::Approx(0.35f).scale(0).epsilon(1e-6));
    REQUIRE(s.emissives.size() == 1);
    CHECK(s.emissives[0].radiance.x == 40.0f);

    // The glowing sphere is the first shape, and the one light: a record of
    // kind sphere into the sphere lights.
    REQUIRE(s.lights.size() == 1);
    CHECK(s.lights[0].kind == lights::LightKind::sphere);
    CHECK(s.lights[0].index == 0);
    CHECK(s.light_counts.lights == 1);
    REQUIRE(s.sphere_lights.size() == 1);
    CHECK(s.light_counts.spheres == 1);
    // Which light each shape is: the glow is, the brass and the example's
    // own shapes are not.
    REQUIRE(s.shape_lights.size() == s.shapes.records.size());
    CHECK(s.shape_lights[0] == 0);
    for (std::size_t i = 1; i < s.shape_lights.size(); ++i) {
        CHECK(s.shape_lights[i] == lights::no_light);
    }
    // The light is its shape: where it is and how big are the shape's.
    CHECK(s.sphere_lights[0].shape == 0);
    CHECK(s.shapes.transforms[0].m[2][3] == 3.0f);
    CHECK(s.shapes.transforms[0].m[0][0] == doctest::Approx(0.05f).scale(0).epsilon(1e-6));
    CHECK(s.sphere_lights[0].radiance.y == 36.0f);
}

TEST_CASE("every mistake is refused, naming the file and the line") {
    CHECK(contains(error_of(with("radius = 0.8", "radius = 0")), "s.toml:30: shape 1's radius must be greater than 0"));
    CHECK(contains(error_of(with("ior = 1.5", "ior = 1")), "s.toml:21: material 'glass''s ior must be greater than 1"));
    CHECK(contains(error_of(with("size = 0.5", "size = -1")), "s.toml:15: "));
    CHECK(contains(error_of(with("size = 0.5", "size = -1")), "size must be greater than 0"));
    CHECK(contains(error_of(with("max = [6, 0, 6]", "max = [6, -0.1, 6]")), "s.toml:"));
    CHECK(contains(error_of(with("max = [6, 0, 6]", "max = [6, -0.1, 6]")), "min must be below its max"));
    CHECK(contains(error_of(with("look_at = [0, 0.8, 0]", "look_at = [0, 1.2, 4]")), "s.toml:2: "));
    CHECK(contains(error_of(with("look_at = [0, 0.8, 0]", "look_at = [0, 1.2, 4]")), "cannot be framed"));
    CHECK(contains(error_of(with("vertical_fov_degrees = 40", "vertical_fov_degrees = 180")), "s.toml:2: "));
    CHECK(contains(error_of(with("vertical_fov_degrees = 40", "vertical_fov_degrees = 180")), "cannot be framed"));
    CHECK(contains(error_of(with("kind = \"sphere\"", "kind = \"cone\"")), "s.toml:28: unknown shape kind 'cone'"));
    CHECK(contains(error_of(with("kind = \"dielectric\"", "kind = \"metal\"")),
                   "s.toml:20: unknown material kind 'metal'"));
    CHECK(contains(error_of(with("kind = \"checker\"", "kind = \"marble\"")),
                   "s.toml:14: unknown texture kind 'marble'; known kinds: checker, wood"));
    CHECK(contains(error_of(with("kind = \"gradient\"", "kind = \"stars\"")),
                   "s.toml:9: unknown environment kind 'stars'"));
    CHECK(contains(error_of(with("material = \"glass\"", "material = \"crystal\"")),
                   "s.toml:31: shape 1 uses material 'crystal', which is not defined"));
    CHECK(contains(error_of(with("texture = \"floor_checks\"", "texture = \"tiles\"")),
                   "s.toml:25: material 'floor' uses texture 'tiles', which is not defined"));
    CHECK(contains(error_of(with("texture = \"floor_checks\"", "texture = \"floor_checks\"\ncolor = [1, 1, 1]")),
                   "exactly one of 'color' and 'texture'"));
    CHECK(contains(error_of(with("radius = 0.8", "radius = 0.8\nshiny = true")),
                   "s.toml:31: unknown key 'shiny' in shape 1"));
    CHECK(contains(error_of(with("[camera]", "[camera]\nzoom = 2")), "s.toml:3: unknown key 'zoom' in [camera]"));
    CHECK(contains(error_of(with("[environment]", "[lights]\n[environment]")),
                   "s.toml:8: unknown key 'lights' in the scene"));
    CHECK(contains(error_of(with("radius = 0.8", "radius = \"big\"")), "s.toml:30: "));
    CHECK(contains(error_of(with("radius = 0.8", "radius = \"big\"")), "must be a number"));
    // Finite as a double, beyond float's range: refused, not narrowed to infinity.
    CHECK(contains(error_of(with("radius = 0.8", "radius = 1e300")), "s.toml:30: "));
    CHECK(contains(error_of(with("radius = 0.8", "radius = 1e300")), "within float's range"));
    CHECK(contains(error_of(with("ior = 1.5", "ior = 1e39")), "s.toml:21: "));
    CHECK(contains(error_of(with("ior = 1.5", "ior = 1e39")), "within float's range"));
    CHECK(contains(error_of(with("center = [0, 0.8, 0]", "center = [0, -1e300, 0]")), "s.toml:29: "));
    CHECK(contains(error_of(with("center = [0, 0.8, 0]", "center = [0, -1e300, 0]")), "within float's range"));
    CHECK(contains(error_of(with("center = [0, 0.8, 0]", "center = [0, 0.8]")), "s.toml:29: "));
    CHECK(contains(error_of(with("center = [0, 0.8, 0]", "center = [0, 0.8]")), "array of three numbers"));
    CHECK(contains(error_of(with("radius = 0.8\n", "")), "s.toml:27: "));
    CHECK(contains(error_of(with("radius = 0.8\n", "")), "shape 1 has no 'radius'"));
    CHECK(contains(error_of("[camera]\nposition = [0, 0, 1]\nlook_at = [0, 0, 0]\nvertical_fov_degrees = 40\n"
                            "[environment]\nkind = \"gradient\"\nzenith = [0, 0, 0]\nhorizon = [0, 0, 0]\n"),
                   "no 'shapes'"));
    CHECK(contains(error_of("[camera\n"), "s.toml:1:"));

    const std::string brass = "[materials.brass]\nkind = \"conductor\"\n";
    // [materials.brass] lands on line 27, its f0 on 29, its roughness on 30.
    CHECK(contains(error_of(with("[[shapes]]", brass + "f0 = [1.2, 0.8, 0.4]\nroughness = 0.3\n[[shapes]]")),
                   "s.toml:29: "));
    CHECK(contains(error_of(with("[[shapes]]", brass + "f0 = [1.2, 0.8, 0.4]\nroughness = 0.3\n[[shapes]]")),
                   "f0 must be within [0, 1]"));
    CHECK(contains(error_of(with("[[shapes]]", brass + "f0 = [0.9, 0.8, 0.4]\nroughness = 0\n[[shapes]]")),
                   "s.toml:30: "));
    CHECK(contains(error_of(with("[[shapes]]", brass + "f0 = [0.9, 0.8, 0.4]\nroughness = 0\n[[shapes]]")),
                   "roughness must be in (0, 1]"));
    const std::string glow = "[materials.glow]\nkind = \"emissive\"\n";
    CHECK(contains(error_of(with("[[shapes]]", glow + "radiance = [1, -1, 1]\n[[shapes]]")), "s.toml:29: "));
    CHECK(contains(error_of(with("[[shapes]]", glow + "radiance = [1, -1, 1]\n[[shapes]]")),
                   "radiance must not be negative"));
    std::string glowing_box = with("[[shapes]]", glow + "radiance = [1, 1, 1]\n[[shapes]]");
    const std::string floor_material = "material = \"floor\"";
    glowing_box.replace(glowing_box.rfind(floor_material), floor_material.size(), "material = \"glow\"");
    CHECK(contains(error_of(glowing_box), "only a sphere may be emissive"));
}

namespace {

// The example with a third sphere, which wanders, clear of the others.
std::string moving(const std::string& center = "[3, 1.5, 0]",
                   const std::string& motion = "{ kind = \"wander\", reach = 0.25, speed = 0.3, seed = 7 }",
                   const std::string& radius = "0.05") {
    return std::string(example) + "\n[[shapes]]\nkind = \"sphere\"\ncenter = " + center + "\nradius = " + radius +
           "\nmaterial = \"glass\"\nmotion = " + motion + "\n";
}

}  // namespace

TEST_CASE("a box keeps the file's corners exactly, where a center and a half extent would not") {
    // 999999.9375 to 1000000, adjacent floats at the world's edge: their
    // middle, 999999.96875, is not a float, and a center and half extent
    // would round it and give back other corners.
    const std::string text =
        with("min = [-6, -0.1, -6]\nmax = [6, 0, 6]", "min = [999999.9375, -0.1, -6]\nmax = [1000000, 0, 6]");
    const scene::SceneDescription far = scene::parse(text, "s");
    const shapes::Bounds placed =
        shapes::world_bounds(shapes::object_bounds(far.shapes, far.shapes.records[1]), far.shapes.transforms[1]);
    CHECK(placed.min.x == 999999.9375f);
    CHECK(placed.max.x == 1000000.0f);
}

TEST_CASE("a sphere's motion: a mover for its shape, its wander about its center") {
    const scene::SceneDescription s = scene::parse(moving(), "s");
    REQUIRE(s.shapes.records.size() == 3);
    CHECK(animation::moves(s.animation));
    REQUIRE(s.animation.movers.size() == 1);
    CHECK(s.animation.movers[0].target == 2);
    CHECK(s.animation.movers[0].motion.kind == animation::MotionKind::wander);
    REQUIRE(s.animation.motions.wanders.size() == 1);
    const animation::Wander& w = s.animation.motions.wanders[0];
    CHECK(w.anchor.x == 3.0f);
    CHECK(w.anchor.y == 1.5f);
    CHECK(w.reach == 0.25f);
    // The shape at rest is at the anchor.
    CHECK(s.shapes.transforms[2].m[0][3] == 3.0f);
}

TEST_CASE("the wandering brass sphere scene reads, both fireflies moving") {
    const scene::SceneDescription s = scene::load(tests::scenes_dir / "brass_sphere_wander.toml");
    REQUIRE(s.animation.movers.size() == 2);
    CHECK(s.animation.movers[0].target == 2);
    CHECK(s.animation.movers[1].target == 3);
    CHECK(s.sphere_lights.size() == 2);
}

TEST_CASE("every mistake in a motion is refused, naming the file and the line") {
    // The third sphere's motion is on line 44.
    const auto motion = [](const std::string& m) { return error_of(moving("[3, 1.5, 0]", m)); };
    CHECK(contains(motion("{ kind = \"orbit\", reach = 0.25, speed = 0.3, seed = 7 }"),
                   "s.toml:44: unknown motion kind 'orbit'"));
    CHECK(contains(motion("{ kind = \"wander\", reach = 0, speed = 0.3, seed = 7 }"),
                   "s.toml:44: shape 3's motion's reach must be greater than 0"));
    CHECK(contains(motion("{ kind = \"wander\", reach = 0.25, speed = -1, seed = 7 }"),
                   "speed must be greater than 0"));
    CHECK(contains(motion("{ kind = \"wander\", reach = 0.25, speed = 0.3, seed = -1 }"),
                   "seed must be an integer, 0 or more"));
    CHECK(contains(motion("{ kind = \"wander\", reach = 0.25, speed = 0.3, seed = 1.5 }"),
                   "seed must be an integer, 0 or more"));
    CHECK(contains(motion("{ kind = \"wander\", reach = 0.25, speed = 0.3 }"), "s.toml:44: "));
    CHECK(contains(motion("{ kind = \"wander\", reach = 0.25, speed = 0.3 }"), "has no 'seed'"));
    CHECK(contains(motion("{ kind = \"wander\", reach = 0.25, speed = 0.3, seed = 7, size = 1 }"),
                   "s.toml:44: unknown key 'size' in shape 3's motion"));
    CHECK(contains(motion("\"wander\""), "s.toml:44: shape 3's motion must be a table"));
    // Only a sphere moves.
    CHECK(contains(error_of(with("material = \"floor\"", "material = \"floor\"\nmotion = { kind = \"wander\" }")),
                   "s.toml:38: unknown key 'motion' in shape 2"));
    // Wherever it wanders, grown by its radius, it may touch no still shape:
    // the glass sphere (0.6 from its center, radius 0.8), or the floor.
    const std::string near_glass = error_of(moving("[0.9, 0.8, 0]"));
    CHECK(contains(near_glass, "s.toml:"));
    CHECK(contains(near_glass, "shape 3's motion: its wander could carry it into a still shape"));
    CHECK(contains(error_of(moving("[3, 0.2, 0]")), "its wander could carry it into a still shape"));
    // Just clear of the floor: 0.25 + 0.05 above its top, at 0.
    CHECK_NOTHROW((void)scene::parse(moving("[3, 0.3001, 0]"), "s"));
    // Out of the world: the shape itself, and then where its wander could
    // carry it, grown by its radius.
    CHECK(contains(error_of(moving("[3e38, 1.5, 0]")), "shape 3 must lie within 1000 km of the origin"));
    CHECK(contains(error_of(moving("[999999.8, 1.5, 0]")),
                   "shape 3's motion could carry shape 3 out of the world: it must stay within 1000 km"));
    CHECK_NOTHROW((void)scene::parse(moving("[999999.5, 1.5, 0]"), "s"));
}

TEST_CASE("moving shapes are not checked against each other") {
    const std::string both = moving() +
                             "\n[[shapes]]\nkind = \"sphere\"\ncenter = [3.1, 1.5, 0]\nradius = 0.05\n"
                             "material = \"glass\"\nmotion = { kind = \"wander\", reach = 0.25, speed = 0.3, "
                             "seed = 8 }\n";
    const scene::SceneDescription s = scene::parse(both, "s");
    CHECK(s.animation.movers.size() == 2);
}

TEST_CASE("the flying brass sphere scene reads: six flights, each with its flight's flashes") {
    constexpr std::size_t flights = 6;
    const scene::SceneDescription s = scene::load(tests::scenes_dir / "brass_sphere_flight.toml");
    CHECK(s.animation.motions.flights.size() == flights);
    REQUIRE(s.animation.glowers.size() == flights);
    for (std::size_t i = 0; i < flights; ++i) {
        CHECK(s.animation.glowers[i].target == i);  // sphere light i
        CHECK(s.animation.glowers[i].glow.kind == animation::GlowKind::schedule);
        CHECK(s.animation.glows.schedules[i].schedule.starts == s.animation.motions.flights[i].flashes.starts);
    }
    CHECK(scene::changes(s));
}

TEST_CASE("names, and glows: what they read into, and every mistake refused") {
    const std::string glow = "[materials.light]\nkind = \"emissive\"\nradiance = [5, 5, 5]\n";
    const auto lamp = [&](const std::string& extra) {
        return with("[[shapes]]", glow + "[[shapes]]\nkind = \"sphere\"\ncenter = [3, 1, 0]\nradius = 0.05\n"
                                         "material = \"light\"\n" +
                                      extra + "\n[[shapes]]");
    };
    // A still light that blinks in a rhythm: a glower for sphere light 0.
    const scene::SceneDescription s =
        scene::parse(lamp("glow = { kind = \"rhythm\", period = 5, flash = 0.4, dim = 0.1, seed = 2 }"), "s");
    REQUIRE(s.animation.glowers.size() == 1);
    CHECK(s.animation.glowers[0].target == 0);
    CHECK(s.animation.glows.rhythms[0].period == 5.0);
    CHECK(scene::changes(s));
    CHECK_FALSE(animation::moves(s.animation));

    // The lamp's glow is on line 35.
    CHECK(contains(error_of(lamp("glow = { kind = \"rhythm\", period = 5, flash = 3, dim = 0.1, seed = 2 }")),
                   "s.toml:35: "));
    CHECK(contains(error_of(lamp("glow = { kind = \"rhythm\", period = 5, flash = 3, dim = 0.1, seed = 2 }")),
                   "at most half its period"));
    CHECK(contains(error_of(lamp("glow = { kind = \"rhythm\", period = 5, flash = 0.4, dim = 1, seed = 2 }")),
                   "dim must be in [0, 1)"));
    CHECK(contains(error_of(lamp("glow = { kind = \"flight\", flash = 0.4, dim = 0.1 }")), "does not fly"));
    CHECK(contains(error_of(lamp("glow = { kind = \"candle\" }")), "s.toml:35: unknown glow kind 'candle'"));
    // A period too short to count flashes in exactly for any t (glow.h).
    CHECK(contains(
        error_of(lamp("glow = { kind = \"rhythm\", period = 0.0001, flash = 0.00004, dim = 0.1, seed = 2 }")),
        "period must be at least 0.001 s"));
    CHECK_NOTHROW((void)scene::parse(
        lamp("glow = { kind = \"rhythm\", period = 0.001, flash = 0.0004, dim = 0.1, seed = 2 }"), "s"));
    CHECK(contains(error_of(with("radius = 0.8", "radius = 0.8\nglow = { kind = \"rhythm\", period = 5, "
                                                 "flash = 0.4, dim = 0.1, seed = 2 }")),
                   "s.toml:31: "));
    CHECK(contains(error_of(with("radius = 0.8", "radius = 0.8\nglow = { kind = \"rhythm\", period = 5, "
                                                 "flash = 0.4, dim = 0.1, seed = 2 }")),
                   "has a glow, and is not a light"));
    // Names: unique, and only on shapes.
    const std::string twice =
        replaced(with("radius = 0.8", "radius = 0.8\nname = \"a\""), "material = \"floor\"",
                 "name = \"a\"\nmaterial = \"floor\"");
    CHECK(contains(error_of(twice), "the name 'a' is used twice"));
}

TEST_CASE("a missing file is refused by name") {
    CHECK_THROWS_WITH_AS(scene::load("no/such/scene.toml"), doctest::Contains("no/such/scene.toml"), scene::Error);
}

TEST_CASE("a wood texture reads as the table's planks, every value checked") {
    const std::string wood = "[textures.walnut]\nkind = \"wood\"\nlight = [0.13, 0.065, 0.03]\n"
                             "dark = [0.045, 0.02, 0.008]\nring = 0.004\nboard = 0.16\nseed = 3\n";
    // The example with the walnut before the glass, on lines 19 to 25, and
    // the floor made of it.
    const auto with_wood = [&](std::string_view from, std::string_view to) {
        const std::string text = from.empty() ? wood : replaced(wood, from, to);
        return replaced(with("[materials.glass]", text + "[materials.glass]"), "texture = \"floor_checks\"",
                        "texture = \"walnut\"");
    };
    const scene::SceneDescription s = scene::parse(with_wood("", ""), "s");
    REQUIRE(s.woods.size() == 1);
    REQUIRE(s.textures.size() == 2);  // floor_checks, walnut: name order
    CHECK(s.textures[1].kind == textures::TextureKind::wood);
    CHECK(s.textures[1].index == 0);
    CHECK(s.rough[0].texture.index == 1);
    CHECK(s.woods[0].light.y == doctest::Approx(0.065f).scale(0).epsilon(1e-6));
    CHECK(s.woods[0].dark.z == doctest::Approx(0.008f).scale(0).epsilon(1e-6));
    CHECK(s.woods[0].ring == 0.004f);
    CHECK(s.woods[0].board == 0.16f);
    CHECK(s.woods[0].seed == 3u);

    const auto error = [&](std::string_view from, std::string_view to) { return error_of(with_wood(from, to)); };
    CHECK(contains(error("light = [0.13, 0.065, 0.03]", "light = [1.2, 0.065, 0.03]"),
                   "s.toml:21: texture 'walnut''s light must be within [0, 1]"));
    CHECK(contains(error("dark = [0.045, 0.02, 0.008]", "dark = [0.045, -0.1, 0.008]"),
                   "s.toml:22: texture 'walnut''s dark must be within [0, 1]"));
    CHECK(contains(error("ring = 0.004", "ring = 0.00005"),
                   "s.toml:23: texture 'walnut''s ring must be at least 0.0001 m"));
    CHECK(contains(error("ring = 0.004", "ring = 1e-45"), "ring must be at least 0.0001 m"));
    CHECK(contains(error("board = 0.16", "board = 0.0005"),
                   "s.toml:24: texture 'walnut''s board must be from 0.001 to 10 m"));
    CHECK(contains(error("board = 0.16", "board = 11"), "board must be from 0.001 to 10 m"));
    CHECK(contains(error("seed = 3", "seed = 4294967296"),
                   "s.toml:25: texture 'walnut''s seed must be at most 4294967295"));
    CHECK(contains(error("seed = 3", "seed = -1"), "seed must be an integer, 0 or more"));
    CHECK(contains(error("seed = 3", "seed = 3\ngrain = 1"), "s.toml:26: unknown key 'grain' in texture 'walnut'"));
    CHECK(contains(error("ring = 0.004\n", ""), "s.toml:19: texture 'walnut' has no 'ring'"));
}

TEST_CASE("every shape lies within the world") {
    CHECK(contains(error_of(with("center = [0, 0.8, 0]", "center = [999999.5, 0.8, 0]")),
                   "shape 1 must lie within 1000 km of the origin on every axis"));
    CHECK(contains(error_of(with("max = [6, 0, 6]", "max = [6, 0, 1000001]")),
                   "shape 2 must lie within 1000 km of the origin on every axis"));
    CHECK_NOTHROW((void)scene::parse(with("max = [6, 0, 6]", "max = [6, 0, 1000000]"), "s"));
}

TEST_CASE("the marbles' scene reads: a walnut table, thirty-six marbles, 616 fireflies") {
    constexpr std::size_t fireflies = 480 + 136;  // two swarms'
    const scene::SceneDescription s = scene::load(tests::scenes_dir / "marbles.toml");
    CHECK(s.woods.size() == 1);
    CHECK(s.swirls.size() == 5);
    CHECK(s.coated.size() == 7);
    CHECK(s.media.size() == 6);
    CHECK(s.absorbing.size() == 6);
    // The table's top, legs and the ground; thirty-six marbles, nine cores,
    // and the fireflies.
    CHECK(s.shapes.records.size() == 6 + 36 + 9 + fireflies);
    std::size_t filled = 0;
    for (const shapes::ShapeRecord& record : s.shapes.records) {
        filled += record.interior != contracts::no_medium ? 1u : 0u;
    }
    CHECK(filled == 9);
    CHECK(s.sphere_lights.size() == fireflies);
    CHECK(s.animation.movers.size() == fireflies);
    CHECK(s.animation.glowers.size() == fireflies);
    CHECK(s.animation.motions.flights.size() == fireflies);
}

TEST_CASE("a camera's lens reads, and a camera without one is a pinhole") {
    const scene::SceneDescription pinhole = scene::parse(example, "s");
    CHECK(pinhole.camera.lens_radius == 0.0f);
    const std::string with_lens = "vertical_fov_degrees = 40\nlens = { radius = 0.012, focus = 4 }";
    const scene::SceneDescription lensed = scene::parse(with("vertical_fov_degrees = 40", with_lens), "s");
    CHECK(lensed.camera.lens_radius == 0.012f);
    CHECK(lensed.camera.focus_distance == 4.0f);
    // The lens is on line 7.
    const auto lens = [](const std::string& value) {
        return error_of(with("vertical_fov_degrees = 40", "vertical_fov_degrees = 40\nlens = " + value));
    };
    CHECK(contains(lens("{ radius = -0.01, focus = 4 }"), "s.toml:7: the camera's lens's radius must be 0 or more"));
    CHECK(contains(lens("{ radius = 0.01, focus = 0 }"), "s.toml:7: the camera's lens's focus must be greater than 0"));
    CHECK(contains(lens("{ radius = 0.01 }"), "s.toml:7: the camera's lens has no 'focus'"));
    CHECK(contains(lens("{ radius = 0.01, focus = 4, blades = 6 }"),
                   "s.toml:7: unknown key 'blades' in the camera's lens"));
    CHECK(contains(lens("0.01"), "s.toml:7: the camera's lens must be a table"));
}

TEST_CASE("coated, swirl and media read, each mistake refused at its line") {
    // Before the example's first shape, from line 27: the swirl on 28 to 34,
    // the porcelain on 35 to 38, the core on 39 to 41, the tint on 42 to 45.
    const std::string extra = R"(
[textures.cats_eye]
kind = "swirl"
a = [0.9, 0.35, 0.05]
b = [0.95, 0.9, 0.8]
vanes = 3
twist = 0.5
seed = 2
[materials.porcelain]
kind = "coated"
color = [0.6, 0.05, 0.04]
ior = 1.5
[materials.core]
kind = "rough"
texture = "cats_eye"
[media.blue_tint]
kind = "absorbing"
tint = [0.25, 0.5, 0.95]
tint_distance = 0.01
)";
    const auto scene_with = [&](std::string_view from, std::string_view to, std::string_view shape_extra) {
        std::string text = with("[[shapes]]", extra + "[[shapes]]");
        if (!from.empty()) {
            text = replaced(text, from, to);
        }
        // The glass sphere, filled.
        return replaced(text, "material = \"glass\"", "material = \"glass\"" + std::string(shape_extra));
    };
    const scene::SceneDescription s = scene::parse(scene_with("", "", "\ninterior = \"blue_tint\""), "s");
    REQUIRE(s.coated.size() == 1);
    CHECK(s.coated[0].ior == 1.5f);
    CHECK(s.coated[0].escape ==
          doctest::Approx(static_cast<float>(materials::internal_escape(1.5))).scale(0).epsilon(1e-6));
    REQUIRE(s.swirls.size() == 1);
    CHECK(s.swirls[0].vanes == 3u);
    CHECK(s.swirls[0].twist == 0.5f);
    REQUIRE(s.absorbing.size() == 1);
    // absorption = -ln(tint) / tint_distance.
    CHECK(s.absorbing[0].absorption.x == doctest::Approx(-std::log(0.25) / 0.01).scale(0).epsilon(1e-6));
    CHECK(s.absorbing[0].absorption.z == doctest::Approx(-std::log(0.95) / 0.01).scale(0).epsilon(1e-6));
    CHECK(s.shapes.records[0].interior == 0u);                     // the glass sphere, filled
    CHECK(s.shapes.records[1].interior == contracts::no_medium);  // the floor
    // Without an interior, air.
    CHECK(scene::parse(scene_with("", "", ""), "s").shapes.records[0].interior == contracts::no_medium);

    const auto error = [&](std::string_view from, std::string_view to, std::string_view shape_extra = "") {
        return error_of(scene_with(from, to, shape_extra));
    };
    CHECK(contains(error("color = [0.6, 0.05, 0.04]", "color = [1.2, 0.05, 0.04]"),
                   "s.toml:37: material 'porcelain''s color must be within [0, 1]"));
    CHECK(contains(error("ior = 1.5\n[materials.core]", "ior = 1\n[materials.core]"),
                   "s.toml:38: material 'porcelain''s ior must be greater than 1"));
    CHECK(contains(error("vanes = 3", "vanes = 0"),
                   "s.toml:32: texture 'cats_eye''s vanes must be an integer from 1 to 16"));
    CHECK(contains(error("vanes = 3", "vanes = 17"), "vanes must be an integer from 1 to 16"));
    CHECK(contains(error("a = [0.9, 0.35, 0.05]", "a = [0.9, 1.35, 0.05]"),
                   "s.toml:30: texture 'cats_eye''s a must be within [0, 1]"));
    CHECK(contains(error("tint = [0.25, 0.5, 0.95]", "tint = [0, 0.5, 0.95]"), "s.toml:44: "));
    CHECK(contains(error("tint = [0.25, 0.5, 0.95]", "tint = [0, 0.5, 0.95]"), "tint must be within (0, 1]"));
    CHECK(contains(error("tint = [0.25, 0.5, 0.95]", "tint = [0.25, 1.5, 0.95]"), "tint must be within (0, 1]"));
    CHECK(contains(error("tint_distance = 0.01", "tint_distance = 0"), "s.toml:45: "));
    CHECK(contains(error("tint_distance = 0.01", "tint_distance = 0"), "tint_distance must be greater than 0"));
    CHECK(contains(error("tint = [0.25, 0.5, 0.95]\ntint_distance = 0.01",
                         "tint = [1e-30, 0.5, 0.95]\ntint_distance = 1e-38"),
                   "absorbs past what a float holds"));
    CHECK(contains(error("kind = \"absorbing\"", "kind = \"smoky\""), "s.toml:43: unknown medium kind 'smoky'"));
    CHECK(contains(error("", "", "\ninterior = \"red_tint\""), "interior is medium 'red_tint', which is not defined"));
    // A medium fills only what light passes into: not the floor.
    const std::string floor_filled =
        replaced(scene_with("", "", ""), "material = \"floor\"", "material = \"floor\"\ninterior = \"blue_tint\"");
    CHECK(contains(error_of(floor_filled), "shape 2 has an interior, and light cannot pass into it"));
    // A coated surface's checker must stay within [0, 1] as its color would.
    const std::string bright_checks = replaced(
        with("[[shapes]]",
             extra + "[materials.tiled]\nkind = \"coated\"\ntexture = \"floor_checks\"\nior = 1.5\n[[shapes]]"),
        "a = [0.9, 0.9, 0.9]", "a = [1.9, 0.9, 0.9]");
    CHECK(contains(error_of(bright_checks), "material 'tiled''s texture's colors must be within [0, 1]"));
}

TEST_CASE("a scene file that cannot be read whole is refused, never read as an empty scene") {
    // A directory opens, and reads nothing: once parsed as an empty scene and
    // reported as one with no camera; now refused as unreadable.
    CHECK_THROWS_WITH_AS(scene::load("scenes"), doctest::Contains("cannot read scene file scenes"), scene::Error);
}
