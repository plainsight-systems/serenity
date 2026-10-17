// Whether a scene's materials fit contract 2's packed form
// (core/scene/packed_range.h): the endpoints taken, and every value past
// them refused by the material's name; the scene reader itself unchanged.

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include <doctest/doctest.h>

#include "core/contracts/bsdf.h"
#include "core/scene/packed_range.h"
#include "core/scene/scene.h"
#include "support/scene_text.h"
#include "support/text.h"

using namespace serenity;
using tests::contains;
using tests::replaced;

namespace {

// core/scene/scene.h's example, a glass marble on a floor of checkers, with
// a matte ball, a brass ball and a coated tile beside it: every kind that
// packs a value the reader leaves open.
std::string example() {
    return replaced(tests::scene_example, "\n[[shapes]]", R"(
[materials.paint]
kind = "rough"
color = [0.5, 0.5, 0.5]

[materials.tile]
kind = "coated"
color = [0.4, 0.2, 0.1]
ior = 1.5

[materials.brass]
kind = "conductor"
f0 = [0.9, 0.7, 0.3]
roughness = 0.3

[[shapes]])") + R"(
[[shapes]]
kind = "sphere"
center = [1, 0.3, 0]
radius = 0.3
material = "paint"

[[shapes]]
kind = "sphere"
center = [-1, 0.3, 0]
radius = 0.3
material = "tile"

[[shapes]]
kind = "sphere"
center = [2, 0.3, 0]
radius = 0.3
material = "brass"
)";
}

std::optional<std::string> outside(const std::string& text) {
    return scene::outside_packed_bsdf(scene::parse(text, "s.toml"));
}

}  // namespace

TEST_CASE("a scene whose materials fit the packed form is taken, up to its endpoints") {
    CHECK_FALSE(outside(example()).has_value());
    // Half's largest, in each value the reader leaves open; and values below
    // half's least, which the contract holds to 2^-24 absolutely.
    CHECK_FALSE(outside(replaced(example(), "ior = 1.5\n\n[materials.floor]", "ior = 65504\n\n[materials.floor]"))
                    .has_value());
    CHECK_FALSE(outside(replaced(example(), "ior = 1.5\n\n[materials.brass]", "ior = 65504\n\n[materials.brass]"))
                    .has_value());
    CHECK_FALSE(outside(replaced(example(), "color = [0.5, 0.5, 0.5]", "color = [65504, -65504, 1e-30]")).has_value());
    CHECK_FALSE(outside(replaced(example(), "a = [0.9, 0.9, 0.9]", "a = [65504, 0, 1e-9]")).has_value());
    // The reader records each material's name, in record order, which a
    // refusal names.
    const scene::SceneDescription s = scene::parse(example(), "s.toml");
    REQUIRE(s.material_names.size() == s.materials.size());
    CHECK(s.material_names.front() == "brass");  // name order (scene.h)
}

TEST_CASE("a value past the packed form's largest is refused, naming its material and the value") {
    const auto refused = [](const std::string& text, const std::string& naming) {
        const std::optional<std::string> found = outside(text);
        REQUIRE(found.has_value());
        CHECK(contains(*found, naming));
        CHECK(contains(*found, "at most 65504"));
    };
    // The review's case: an ior of 70000, valid for the reader, would pack
    // to infinity. And the least float past 65504.
    refused(replaced(example(), "ior = 1.5\n\n[materials.floor]", "ior = 70000\n\n[materials.floor]"),
            "material 'glass''s ior, 70000,");
    refused(replaced(example(), "ior = 1.5\n\n[materials.floor]", "ior = 65504.004\n\n[materials.floor]"),
            "material 'glass''s ior, 65504.0039,");
    refused(replaced(example(), "ior = 1.5\n\n[materials.brass]", "ior = 1e5\n\n[materials.brass]"),
            "material 'tile''s ior, 100000,");
    refused(replaced(example(), "color = [0.5, 0.5, 0.5]", "color = [0.5, -1e5, 0.5]"), "material 'paint''s color");
    refused(replaced(example(), "b = [0.1, 0.1, 0.1]", "b = [0.1, 0.1, 7e4]"), "material 'floor''s texture's colors");
    // The scene reader takes every one of them, as it did: the path graph,
    // which packs nothing, renders them (metal/frame/renderer.h).
    CHECK_NOTHROW(scene::parse(
        replaced(example(), "ior = 1.5\n\n[materials.floor]", "ior = 70000\n\n[materials.floor]"), "s.toml"));
}

TEST_CASE("a description built in code is checked by every kind, and names a material by its index") {
    scene::SceneDescription s = scene::parse(example(), "s.toml");
    s.material_names.clear();
    std::size_t conductors = 0;
    std::size_t coateds = 0;
    for (std::size_t i = 0; i < s.materials.size(); ++i) {
        if (s.materials[i].kind == materials::MaterialKind::conductor) {
            ++conductors;
            s.conductors.at(s.materials[i].index).f0.y = std::numeric_limits<float>::infinity();
            const std::optional<std::string> found = scene::outside_packed_bsdf(s);
            REQUIRE(found.has_value());
            CHECK(contains(*found, "material " + std::to_string(i) + "'s f0"));
        }
        if (s.materials[i].kind == materials::MaterialKind::coated) {
            ++coateds;
            scene::SceneDescription escaped = scene::parse(example(), "s.toml");
            escaped.coateds.at(s.materials[i].index).escape = std::nanf("");
            const std::optional<std::string> found = scene::outside_packed_bsdf(escaped);
            REQUIRE(found.has_value());
            CHECK(contains(*found, "'tile''s escape"));
        }
    }
    // Each branch ran: the example's one conductor and one coated material.
    CHECK(conductors == 1);
    CHECK(coateds == 1);
}
