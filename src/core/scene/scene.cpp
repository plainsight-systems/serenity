#include "core/scene/scene.h"

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <sstream>

#include <toml++/toml.hpp>

#include "core/animation/wander.h"
#include "core/camera/pinhole.h"

namespace serenity::scene {

namespace {

// Reads one scene file; every error names the file and the line, in the
// form compilers use, so editors can jump to it.
class Reader {
public:
    explicit Reader(std::string_view source) : source_(source) {}

    [[noreturn]] void fail(const toml::source_region& where, const std::string& message) const {
        std::ostringstream s;
        s << source_ << ':' << where.begin.line << ": " << message;
        throw Error(s.str());
    }

    [[noreturn]] void fail(const toml::node& node, const std::string& message) const {
        fail(node.source(), message);
    }

    [[noreturn]] void fail(const std::string& message) const { throw Error(std::string(source_) + ": " + message); }

    // Every key in `table` must be one of `allowed`.
    void only(const toml::table& table, std::initializer_list<std::string_view> allowed, std::string_view what) const {
        for (const auto& [key, value] : table) {
            bool known = false;
            for (std::string_view name : allowed) {
                known = known || key.str() == name;
            }
            if (!known) {
                fail(key.source(), "unknown key '" + std::string(key.str()) + "' in " + std::string(what));
            }
        }
    }

    const toml::node& required(const toml::table& table, std::string_view key, std::string_view what) const {
        const toml::node* node = table.get(key);
        if (node == nullptr) {
            fail(table, std::string(what) + " has no '" + std::string(key) + "'");
        }
        return *node;
    }

    const toml::table& table(const toml::node& node, std::string_view what) const {
        const toml::table* t = node.as_table();
        if (t == nullptr) {
            fail(node, std::string(what) + " must be a table");
        }
        return *t;
    }

    float number(const toml::node& node, std::string_view what) const {
        if (!node.is_number()) {
            fail(node, std::string(what) + " must be a number");
        }
        const std::optional<double> value = node.value<double>();
        // Checked as a double, before narrowing: converting a double outside
        // float's range to float is undefined behavior.
        if (!value || !std::isfinite(*value) || std::abs(*value) > std::numeric_limits<float>::max()) {
            fail(node, std::string(what) + " must be a finite number within float's range");
        }
        return static_cast<float>(*value);
    }

    float number(const toml::table& t, std::string_view key, std::string_view what) const {
        return number(required(t, key, what), std::string(what) + "'s " + std::string(key));
    }

    contracts::Float3 triple(const toml::node& node, std::string_view what) const {
        const toml::array* a = node.as_array();
        if (a == nullptr || a->size() != 3) {
            fail(node, std::string(what) + " must be an array of three numbers");
        }
        return {number((*a)[0], what), number((*a)[1], what), number((*a)[2], what)};
    }

    contracts::Float3 triple(const toml::table& t, std::string_view key, std::string_view what) const {
        return triple(required(t, key, what), std::string(what) + "'s " + std::string(key));
    }

    std::string_view text(const toml::node& node, std::string_view what) const {
        const std::optional<std::string_view> value = node.value<std::string_view>();
        if (!value) {
            fail(node, std::string(what) + " must be a string, in quotes");
        }
        return *value;
    }

    std::string_view text(const toml::table& t, std::string_view key, std::string_view what) const {
        return text(required(t, key, what), std::string(what) + "'s " + std::string(key));
    }

private:
    std::string_view source_;
};

bool below(contracts::Float3 a, contracts::Float3 b) {
    return a.x < b.x && a.y < b.y && a.z < b.z;
}

bool within(contracts::Float3 a, float low, float high) {
    return a.x >= low && a.x <= high && a.y >= low && a.y <= high && a.z >= low && a.z <= high;
}

contracts::Camera read_camera(const Reader& r, const toml::table& t) {
    r.only(t, {"position", "look_at", "up", "vertical_fov_degrees"}, "[camera]");
    contracts::Camera camera{};
    camera.position = r.triple(t, "position", "the camera");
    camera.look_at = r.triple(t, "look_at", "the camera");
    camera.up = t.contains("up") ? r.triple(t, "up", "the camera") : contracts::Float3{0.0f, 1.0f, 0.0f};
    camera.vertical_fov_degrees = r.number(t, "vertical_fov_degrees", "the camera");
    const char* reason = nullptr;
    if (!camera::valid(camera, &reason)) {
        r.fail(t, std::string("the camera cannot be framed: ") + reason);
    }
    return camera;
}

lights::GradientSkyData read_environment(const Reader& r, const toml::table& t) {
    const std::string_view kind = r.text(t, "kind", "[environment]");
    if (kind != "gradient") {
        r.fail(r.required(t, "kind", "[environment]"),
               "unknown environment kind '" + std::string(kind) + "'; known kinds: gradient");
    }
    r.only(t, {"kind", "zenith", "horizon"}, "[environment]");
    lights::GradientSkyData sky{};
    sky.zenith = r.triple(t, "zenith", "[environment]");
    sky.horizon = r.triple(t, "horizon", "[environment]");
    return sky;
}

// Textures, in name order; returns each name's index into the records.
std::map<std::string, std::uint32_t, std::less<>> read_textures(const Reader& r, const toml::table& all,
                                                                SceneDescription& scene) {
    std::map<std::string, std::uint32_t, std::less<>> index;
    for (const auto& [key, node] : all) {
        const std::string name(key.str());
        const std::string what = "texture '" + name + "'";
        const toml::table& t = r.table(node, what);
        const std::string_view kind = r.text(t, "kind", what);
        if (kind == "checker") {
            r.only(t, {"kind", "size", "a", "b"}, what);
            textures::CheckerData checker{};
            checker.size = r.number(t, "size", what);
            if (!(checker.size > 0.0f)) {
                r.fail(r.required(t, "size", what), what + "'s size must be greater than 0");
            }
            checker.a = r.triple(t, "a", what);
            checker.b = r.triple(t, "b", what);
            scene.textures.push_back({textures::TextureKind::checker, static_cast<std::uint32_t>(scene.checkers.size())});
            scene.checkers.push_back(checker);
        } else {
            r.fail(r.required(t, "kind", what), "unknown texture kind '" + std::string(kind) + "'; known kinds: checker");
        }
        index.emplace(name, static_cast<std::uint32_t>(scene.textures.size() - 1));
    }
    return index;
}

// Materials, in name order; returns each name's index into the records.
std::map<std::string, std::uint32_t, std::less<>> read_materials(
    const Reader& r, const toml::table& all, const std::map<std::string, std::uint32_t, std::less<>>& texture_index,
    SceneDescription& scene) {
    std::map<std::string, std::uint32_t, std::less<>> index;
    for (const auto& [key, node] : all) {
        const std::string name(key.str());
        const std::string what = "material '" + name + "'";
        const toml::table& t = r.table(node, what);
        const std::string_view kind = r.text(t, "kind", what);
        if (kind == "rough") {
            r.only(t, {"kind", "color", "texture"}, what);
            const bool has_color = t.contains("color");
            const bool has_texture = t.contains("texture");
            if (has_color == has_texture) {
                r.fail(t, what + " needs exactly one of 'color' and 'texture'");
            }
            materials::RoughData rough{};
            rough.texture.index = contracts::no_texture;
            if (has_color) {
                rough.color = r.triple(t, "color", what);
            } else {
                const std::string_view texture = r.text(t, "texture", what);
                const auto found = texture_index.find(texture);
                if (found == texture_index.end()) {
                    r.fail(r.required(t, "texture", what), what + " uses texture '" + std::string(texture) +
                                                               "', which is not defined");
                }
                rough.texture.index = found->second;
            }
            scene.materials.push_back({materials::MaterialKind::rough, static_cast<std::uint32_t>(scene.rough.size())});
            scene.rough.push_back(rough);
        } else if (kind == "dielectric") {
            r.only(t, {"kind", "ior"}, what);
            materials::DielectricData dielectric{};
            dielectric.ior = r.number(t, "ior", what);
            if (!(dielectric.ior > 1.0f)) {
                r.fail(r.required(t, "ior", what), what + "'s ior must be greater than 1");
            }
            scene.materials.push_back(
                {materials::MaterialKind::dielectric, static_cast<std::uint32_t>(scene.dielectrics.size())});
            scene.dielectrics.push_back(dielectric);
        } else if (kind == "conductor") {
            r.only(t, {"kind", "f0", "roughness"}, what);
            materials::ConductorData conductor{};
            conductor.f0 = r.triple(t, "f0", what);
            if (!within(conductor.f0, 0.0f, 1.0f)) {
                r.fail(r.required(t, "f0", what), what + "'s f0 must be within [0, 1] in each channel");
            }
            conductor.roughness = r.number(t, "roughness", what);
            if (!(conductor.roughness > 0.0f && conductor.roughness <= 1.0f)) {
                r.fail(r.required(t, "roughness", what), what + "'s roughness must be in (0, 1]");
            }
            scene.materials.push_back(
                {materials::MaterialKind::conductor, static_cast<std::uint32_t>(scene.conductors.size())});
            scene.conductors.push_back(conductor);
        } else if (kind == "emissive") {
            r.only(t, {"kind", "radiance"}, what);
            materials::EmissiveData emissive{};
            emissive.radiance = r.triple(t, "radiance", what);
            if (!within(emissive.radiance, 0.0f, std::numeric_limits<float>::max())) {
                r.fail(r.required(t, "radiance", what), what + "'s radiance must not be negative");
            }
            scene.materials.push_back(
                {materials::MaterialKind::emissive, static_cast<std::uint32_t>(scene.emissives.size())});
            scene.emissives.push_back(emissive);
        } else {
            r.fail(r.required(t, "kind", what), "unknown material kind '" + std::string(kind) +
                                                    "'; known kinds: rough, dielectric, conductor, emissive");
        }
        index.emplace(name, static_cast<std::uint32_t>(scene.materials.size() - 1));
    }
    return index;
}

// A shape's motion, as read, for the checks once every shape is known.
struct ReadMotion {
    const toml::node* node;  // the motion's table, for errors
    std::uint32_t shape;
};

animation::MotionRecord read_motion(const Reader& r, const toml::node& node, contracts::Float3 anchor,
                                    std::string_view shape, animation::Motions& motions) {
    const std::string what = std::string(shape) + "'s motion";
    const toml::table& t = r.table(node, what);
    const std::string_view kind = r.text(t, "kind", what);
    if (kind != "wander") {
        r.fail(r.required(t, "kind", what), "unknown motion kind '" + std::string(kind) + "'; known kinds: wander");
    }
    r.only(t, {"kind", "reach", "speed", "seed"}, what);
    const float reach = r.number(t, "reach", what);
    if (!(reach > 0.0f)) {
        r.fail(r.required(t, "reach", what), what + "'s reach must be greater than 0");
    }
    const float speed = r.number(t, "speed", what);
    if (!(speed > 0.0f)) {
        r.fail(r.required(t, "speed", what), what + "'s speed must be greater than 0");
    }
    const toml::node& seed_node = r.required(t, "seed", what);
    const std::optional<std::int64_t> seed = seed_node.is_integer() ? seed_node.value<std::int64_t>() : std::nullopt;
    if (!seed || *seed < 0) {
        r.fail(seed_node, what + "'s seed must be an integer, 0 or more");
    }
    // The float-range rule is checked with the shape's size, after every
    // shape is read (check_motions); make_wander's own check is then met.
    for (int axis = 0; axis < 3; ++axis) {
        const double a = axis == 0 ? anchor.x : (axis == 1 ? anchor.y : anchor.z);
        if (std::abs(a) + static_cast<double>(reach) > std::numeric_limits<float>::max()) {
            r.fail(node, what + " could carry " + std::string(shape) + " out of float's range");
        }
    }
    motions.wanders.push_back(animation::make_wander(anchor, reach, speed, static_cast<std::uint64_t>(*seed)));
    return {animation::MotionKind::wander, static_cast<std::uint32_t>(motions.wanders.size() - 1)};
}

void read_shapes(const Reader& r, const toml::array& all,
                 const std::map<std::string, std::uint32_t, std::less<>>& material_index, SceneDescription& scene,
                 std::vector<ReadMotion>& read_motions) {
    shapes::Shapes& shapes = scene.shapes;
    std::size_t number = 0;
    for (const toml::node& node : all) {
        const std::string what = "shape " + std::to_string(++number);
        const toml::table& t = r.table(node, what);
        const std::string_view kind = r.text(t, "kind", what);
        const auto index = static_cast<std::uint32_t>(shapes.records.size());

        const auto material = [&]() {
            const std::string_view name = r.text(t, "material", what);
            const auto found = material_index.find(name);
            if (found == material_index.end()) {
                r.fail(r.required(t, "material", what),
                       what + " uses material '" + std::string(name) + "', which is not defined");
            }
            return found->second;
        };

        if (kind == "sphere") {
            r.only(t, {"kind", "center", "radius", "material", "motion"}, what);
            const contracts::Float3 center = r.triple(t, "center", what);
            const float radius = r.number(t, "radius", what);
            if (!(radius > 0.0f)) {
                r.fail(r.required(t, "radius", what), what + "'s radius must be greater than 0");
            }
            const std::uint32_t worn_index = material();
            const materials::MaterialRecord worn = scene.materials[worn_index];
            scene.shape_lights.push_back(worn.kind == materials::MaterialKind::emissive
                                             ? static_cast<std::uint32_t>(scene.lights.size())
                                             : lights::no_light);
            if (worn.kind == materials::MaterialKind::emissive) {
                scene.lights.push_back(
                    {lights::LightKind::sphere, static_cast<std::uint32_t>(scene.sphere_lights.size())});
                scene.sphere_lights.push_back({scene.emissives[worn.index].radiance, index});
            }
            shapes.records.push_back({shapes::ShapeKind::sphere, 0u, worn_index, 0u});
            shapes.transforms.push_back(contracts::placed(center, radius));
            if (const toml::node* motion = t.get("motion")) {
                const animation::MotionRecord record =
                    read_motion(r, *motion, center, what, scene.animation.motions);
                scene.animation.movers.push_back({index, record});
                read_motions.push_back({motion, index});
            }
        } else if (kind == "box") {
            r.only(t, {"kind", "min", "max", "material"}, what);
            const contracts::Float3 min = r.triple(t, "min", what);
            const contracts::Float3 max = r.triple(t, "max", what);
            if (!below(min, max)) {
                r.fail(t, what + "'s min must be below its max on every axis");
            }
            const std::uint32_t worn = material();
            if (scene.materials[worn].kind == materials::MaterialKind::emissive) {
                r.fail(r.required(t, "material", what), what + " is a box; only a sphere may be emissive");
            }
            // Its corners as the file gives them, placed by the identity
            // transform (core/shapes/box.h).
            const shapes::BoxData box{min, 0u, max, 0u};
            scene.shape_lights.push_back(lights::no_light);
            shapes.records.push_back(
                {shapes::ShapeKind::box, static_cast<std::uint32_t>(shapes.boxes.size()), worn, 0u});
            shapes.transforms.push_back(contracts::placed({0.0f, 0.0f, 0.0f}, 1.0f));
            shapes.boxes.push_back(box);
        } else {
            r.fail(r.required(t, "kind", what), "unknown shape kind '" + std::string(kind) + "'; known kinds: sphere, box");
        }
    }
    if (shapes.records.empty()) {
        r.fail("the scene has no shapes");
    }
}

// Every moving shape, wherever its motion takes it, must stay within float's
// range and touch no still shape (scene.h).
void check_motions(const Reader& r, const std::vector<ReadMotion>& read_motions, const SceneDescription& scene) {
    const shapes::Shapes& shapes = scene.shapes;
    std::vector<bool> moving(shapes.records.size(), false);
    for (const ReadMotion& m : read_motions) {
        moving[m.shape] = true;
    }
    for (std::size_t i = 0; i < read_motions.size(); ++i) {
        const ReadMotion& m = read_motions[i];
        const animation::Extent extent = animation::extent(scene.animation.motions, scene.animation.movers[i].motion);
        // Only spheres move, and a sphere's size is its scale
        // (core/shapes/sphere.h): the extent grows by it on every side.
        const double radius = shapes.transforms[m.shape].m[0][0];
        const double float_max = std::numeric_limits<float>::max();
        const double lows[3] = {extent.min.x - radius, extent.min.y - radius, extent.min.z - radius};
        const double highs[3] = {extent.max.x + radius, extent.max.y + radius, extent.max.z + radius};
        for (int axis = 0; axis < 3; ++axis) {
            if (lows[axis] < -float_max || highs[axis] > float_max) {
                r.fail(*m.node, "this motion could carry shape " + std::to_string(m.shape + 1) +
                                    " out of float's range");
            }
        }
        // Rounded outward, so the swept box is never smaller than the sweep.
        const auto down = [](double x) {
            const float f = static_cast<float>(x);
            return static_cast<double>(f) > x ? std::nextafter(f, -std::numeric_limits<float>::infinity()) : f;
        };
        const auto up = [](double x) {
            const float f = static_cast<float>(x);
            return static_cast<double>(f) < x ? std::nextafter(f, std::numeric_limits<float>::infinity()) : f;
        };
        const shapes::Bounds swept{{down(lows[0]), down(lows[1]), down(lows[2])},
                                   {up(highs[0]), up(highs[1]), up(highs[2])}};
        for (std::uint32_t other = 0; other < shapes.records.size(); ++other) {
            if (!moving[other] && shapes::touches(shapes, other, swept)) {
                r.fail(*m.node, "shape " + std::to_string(m.shape + 1) + "'s motion could carry it into shape " +
                                    std::to_string(other + 1) + ", which does not move");
            }
        }
    }
}

SceneDescription read_scene(const Reader& r, const toml::table& root) {
    r.only(root, {"camera", "environment", "textures", "materials", "shapes"}, "the scene");
    SceneDescription scene{};

    scene.camera = read_camera(r, r.table(r.required(root, "camera", "the scene"), "[camera]"));
    scene.environment = read_environment(r, r.table(r.required(root, "environment", "the scene"), "[environment]"));

    const toml::table no_entries;
    const toml::table& textures = root.contains("textures") ? r.table(*root.get("textures"), "[textures]") : no_entries;
    const toml::table& materials =
        root.contains("materials") ? r.table(*root.get("materials"), "[materials]") : no_entries;
    const auto texture_index = read_textures(r, textures, scene);
    const auto material_index = read_materials(r, materials, texture_index, scene);

    const toml::node& shapes = r.required(root, "shapes", "the scene");
    const toml::array* list = shapes.as_array();
    if (list == nullptr) {
        r.fail(shapes, "'shapes' must be an array of tables, written [[shapes]]");
    }
    std::vector<ReadMotion> read_motions;
    read_shapes(r, *list, material_index, scene, read_motions);
    check_motions(r, read_motions, scene);
    scene.light_counts.lights = static_cast<std::uint32_t>(scene.lights.size());
    scene.light_counts.spheres = static_cast<std::uint32_t>(scene.sphere_lights.size());
    return scene;
}

}  // namespace

SceneDescription parse(std::string_view text, std::string_view source) {
    toml::table root;
    try {
        root = toml::parse(text, source);
    } catch (const toml::parse_error& error) {
        Reader(source).fail(error.source(), std::string(error.description()));
    }
    return read_scene(Reader(source), root);
}

SceneDescription load(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw Error("cannot read scene file " + path.string());
    }
    std::ostringstream text;
    text << file.rdbuf();
    if (!file && !file.eof()) {
        throw Error("cannot read scene file " + path.string());
    }
    return parse(text.str(), path.string());
}

}  // namespace serenity::scene
