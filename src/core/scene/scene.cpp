#include "core/scene/scene.h"

#include <cmath>
#include <deque>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <sstream>

#include <toml++/toml.hpp>

#include "core/animation/flight.h"
#include "core/animation/glow.h"
#include "core/animation/wander.h"
#include "core/contracts/obstacles.h"
#include "core/camera/thin_lens.h"
#include "core/scene/swarm.h"

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
    r.only(t, {"position", "look_at", "up", "vertical_fov_degrees", "lens"}, "[camera]");
    contracts::Camera camera{};
    camera.position = r.triple(t, "position", "the camera");
    camera.look_at = r.triple(t, "look_at", "the camera");
    camera.up = t.contains("up") ? r.triple(t, "up", "the camera") : contracts::Float3{0.0f, 1.0f, 0.0f};
    camera.vertical_fov_degrees = r.number(t, "vertical_fov_degrees", "the camera");
    if (const toml::node* lens_node = t.get("lens")) {
        const toml::table& lens = r.table(*lens_node, "the camera's lens");
        r.only(lens, {"radius", "focus"}, "the camera's lens");
        camera.lens_radius = r.number(lens, "radius", "the camera's lens");
        if (!(camera.lens_radius >= 0.0f)) {
            r.fail(r.required(lens, "radius", "the camera's lens"), "the camera's lens's radius must be 0 or more");
        }
        camera.focus_distance = r.number(lens, "focus", "the camera's lens");
        if (!(camera.focus_distance > 0.0f)) {
            r.fail(r.required(lens, "focus", "the camera's lens"), "the camera's lens's focus must be greater than 0");
        }
    }
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

std::uint64_t read_seed(const Reader& r, const toml::table& t, const std::string& what) {
    const toml::node& node = r.required(t, "seed", what);
    const std::optional<std::int64_t> seed = node.is_integer() ? node.value<std::int64_t>() : std::nullopt;
    if (!seed || *seed < 0) {
        r.fail(node, what + "'s seed must be an integer, 0 or more");
    }
    return static_cast<std::uint64_t>(*seed);
}

float read_positive(const Reader& r, const toml::table& t, std::string_view key, const std::string& what) {
    const float v = r.number(t, key, what);
    if (!(v > 0.0f)) {
        r.fail(r.required(t, key, what), what + "'s " + std::string(key) + " must be greater than 0");
    }
    return v;
}

float read_at_least_zero(const Reader& r, const toml::table& t, std::string_view key, const std::string& what) {
    const float v = r.number(t, key, what);
    if (!(v >= 0.0f)) {
        r.fail(r.required(t, key, what), what + "'s " + std::string(key) + " must be 0 or more");
    }
    return v;
}

// Whether [lo, hi] lies within the world on every axis (scene.h), in double.
bool inside_world(double lo, double hi) {
    return std::isfinite(lo) && std::isfinite(hi) && lo >= -world_extent && hi <= world_extent;
}

bool inside_world(contracts::Float3 lo, contracts::Float3 hi, double grown) {
    return inside_world(double(lo.x) - grown, double(hi.x) + grown) &&
           inside_world(double(lo.y) - grown, double(hi.y) + grown) &&
           inside_world(double(lo.z) - grown, double(hi.z) + grown);
}

const std::string world_words = "within 1000 km of the origin on every axis";

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
        } else if (kind == "wood") {
            r.only(t, {"kind", "light", "dark", "ring", "board", "seed"}, what);
            textures::WoodData wood{};
            wood.light = r.triple(t, "light", what);
            wood.dark = r.triple(t, "dark", what);
            for (const char* key : {"light", "dark"}) {
                if (!within(r.triple(t, key, what), 0.0f, 1.0f)) {
                    r.fail(r.required(t, key, what), what + "'s " + key + " must be within [0, 1] in every channel");
                }
            }
            wood.ring = r.number(t, "ring", what);
            if (!(wood.ring >= textures::wood_least_ring)) {
                r.fail(r.required(t, "ring", what), what + "'s ring must be at least 0.0001 m");
            }
            wood.board = r.number(t, "board", what);
            if (!(wood.board >= textures::wood_least_board && wood.board <= textures::wood_most_board)) {
                r.fail(r.required(t, "board", what), what + "'s board must be from 0.001 to 10 m");
            }
            const std::uint64_t seed = read_seed(r, t, what);
            if (seed > 0xFFFFFFFFull) {
                r.fail(r.required(t, "seed", what), what + "'s seed must be at most 4294967295");
            }
            wood.seed = static_cast<std::uint32_t>(seed);
            scene.textures.push_back({textures::TextureKind::wood, static_cast<std::uint32_t>(scene.woods.size())});
            scene.woods.push_back(wood);
        } else {
            r.fail(r.required(t, "kind", what),
                   "unknown texture kind '" + std::string(kind) + "'; known kinds: checker, wood");
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

// A swarm as read (swarm.h), and where it was read, for errors.
struct SwarmEntry {
    Swarm swarm;
    const toml::table* table = nullptr;
    std::string what;  // "swarm N"
};

// A shape's motion and glow, as read: made once every shape is read, when
// the still shapes it must keep clear of and the names it circles are known.
// A written shape's are its tables; a swarm's firefly's, its swarm's.
struct Pending {
    std::uint32_t shape;
    std::string what;                    // "shape N", or "swarm N's firefly i", for errors
    const toml::node* motion = nullptr;  // a written shape's motion's table, if any
    const toml::node* glow = nullptr;    // and its glow's
    const SwarmEntry* swarm = nullptr;   // a swarm's firefly: its swarm
    std::uint32_t firefly = 0;           // which of it
    contracts::Float3 start{};           // and where it starts (swarm.h, step 2)
};

// A sphere of `radius` at `center` wearing material `worn_index`, and, if
// that is emissive, a sphere light; its index.
std::uint32_t add_sphere(SceneDescription& scene, contracts::Float3 center, float radius, std::uint32_t worn_index) {
    const auto index = static_cast<std::uint32_t>(scene.shapes.records.size());
    const materials::MaterialRecord worn = scene.materials[worn_index];
    scene.shape_lights.push_back(worn.kind == materials::MaterialKind::emissive
                                     ? static_cast<std::uint32_t>(scene.lights.size())
                                     : lights::no_light);
    if (worn.kind == materials::MaterialKind::emissive) {
        scene.lights.push_back({lights::LightKind::sphere, static_cast<std::uint32_t>(scene.sphere_lights.size())});
        scene.sphere_lights.push_back({scene.emissives[worn.index].radiance, index});
    }
    scene.shapes.records.push_back({shapes::ShapeKind::sphere, 0u, worn_index, 0u});
    scene.shapes.transforms.push_back(contracts::placed(center, radius));
    return index;
}


// The still shapes, as contract 11 asks of them: each shape kind's exact
// distance and touch tests (core/shapes/shapes.h) over every shape that
// does not move.
class StillShapes final : public contracts::Obstacles {
public:
    StillShapes(const shapes::Shapes& shapes, const std::vector<bool>& moving) : shapes_(shapes) {
        for (std::uint32_t i = 0; i < shapes.records.size(); ++i) {
            if (!moving[i]) {
                still_.push_back(i);
            }
        }
    }

    double distance(contracts::Float3 point) const override {
        double nearest = std::numeric_limits<double>::infinity();
        for (std::uint32_t i : still_) {
            nearest = std::min(nearest, shapes::distance(shapes_, i, point));
        }
        return nearest;
    }

    bool touches(const contracts::Box& box) const override {
        const shapes::Bounds bounds{box.min, box.max};
        for (std::uint32_t i : still_) {
            if (shapes::touches(shapes_, i, bounds)) {
                return true;
            }
        }
        return false;
    }

private:
    const shapes::Shapes& shapes_;
    std::vector<std::uint32_t> still_;
};

// A flight's numbers from table `t`, read as a written flight's motion and
// a swarm both write them (scene.h); not yet made.
animation::FlightParams read_flight_params(const Reader& r, const toml::table& t, const std::string& what,
                                           const std::map<std::string, std::uint32_t, std::less<>>& names,
                                           const std::vector<bool>& moving, const SceneDescription& scene) {
    animation::FlightParams params;
    params.volume = {r.triple(t, "min", what), r.triple(t, "max", what)};
    if (!below(params.volume.min, params.volume.max)) {
        r.fail(t, what + "'s min must be below its max on every axis");
    }
    const toml::node& list_node = r.required(t, "targets", what);
    const toml::array* list = list_node.as_array();
    if (list == nullptr) {
        r.fail(list_node, what + "'s targets must be an array of shape names");
    }
    for (const toml::node& name_node : *list) {
        const std::string_view name = r.text(name_node, what + "'s target");
        const auto found = names.find(name);
        if (found == names.end()) {
            r.fail(name_node, what + " circles '" + std::string(name) + "', which is not defined");
        }
        const std::uint32_t target = found->second;
        if (scene.shapes.records[target].kind != shapes::ShapeKind::sphere || moving[target]) {
            r.fail(name_node, what + " circles '" + std::string(name) + "', which is not a still sphere");
        }
        const contracts::Transform& at = scene.shapes.transforms[target];
        params.targets.push_back({contracts::translation(at), at.m[0][0]});
    }
    params.speed = read_positive(r, t, "speed", what);
    params.clearance = read_at_least_zero(r, t, "clearance", what);
    params.weights = {read_at_least_zero(r, t, "circle", what), read_at_least_zero(r, t, "swoop", what),
                      read_at_least_zero(r, t, "drift", what)};
    if (params.weights[0] + params.weights[1] + params.weights[2] <= 0.0f) {
        r.fail(t, what + "'s circle, swoop and drift are all 0");
    }
    if (params.weights[0] > 0.0f && params.targets.empty()) {
        r.fail(r.required(t, "circle", what), what + " circles, and names no targets");
    }
    params.seed = read_seed(r, t, what);
    return params;
}

animation::GlowRecord make_glow(const Reader& r, const Pending& p, const animation::MotionRecord* motion,
                                SceneDescription& scene) {
    const std::string what = p.what + "'s glow";
    const toml::table& t = r.table(*p.glow, what);
    const std::string_view kind = r.text(t, "kind", what);
    const auto dim_of = [&]() {
        const float dim = r.number(t, "dim", what);
        if (!(dim >= 0.0f && dim < 1.0f)) {
            r.fail(r.required(t, "dim", what), what + "'s dim must be in [0, 1)");
        }
        return dim;
    };
    animation::Glows& glows = scene.animation.glows;
    if (kind == "rhythm") {
        r.only(t, {"kind", "period", "flash", "dim", "seed"}, what);
        animation::Rhythm rhythm;
        rhythm.period = read_positive(r, t, "period", what);
        rhythm.flash = read_positive(r, t, "flash", what);
        if (rhythm.flash > rhythm.period / 2.0) {
            r.fail(r.required(t, "flash", what), what + "'s flash must be at most half its period");
        }
        rhythm.dim = dim_of();
        rhythm.seed = read_seed(r, t, what);
        glows.rhythms.push_back(rhythm);
        return {animation::GlowKind::rhythm, static_cast<std::uint32_t>(glows.rhythms.size() - 1)};
    }
    if (kind == "flight") {
        r.only(t, {"kind", "flash", "dim"}, what);
        if (motion == nullptr || motion->kind != animation::MotionKind::flight) {
            r.fail(*p.glow, what + " follows its flight, and " + p.what + " does not fly");
        }
        animation::ScheduleGlow glow;
        glow.schedule = scene.animation.motions.flights[motion->index].flashes;
        glow.flash = read_positive(r, t, "flash", what);
        if (!(glow.flash < 1.0)) {
            r.fail(r.required(t, "flash", what), what + "'s flash must be under a second");
        }
        glow.dim = dim_of();
        glows.schedules.push_back(glow);
        return {animation::GlowKind::schedule, static_cast<std::uint32_t>(glows.schedules.size() - 1)};
    }
    r.fail(r.required(t, "kind", what), "unknown glow kind '" + std::string(kind) + "'; known kinds: rhythm, flight");
}

// Every motion and glow, made now that every shape is read: motions against
// the still shapes, glows from their motions. Wanders are made as they come;
// every flight, written or a swarm's, is gathered and made at once, in
// parallel (core/animation/flight.h, make_flights), then each motion and glow
// is recorded in the order the shapes come.
void make_animation(const Reader& r, const std::vector<Pending>& pending,
                    const std::map<std::string, std::uint32_t, std::less<>>& names, const std::vector<bool>& moving,
                    const contracts::Obstacles& obstacles, SceneDescription& scene) {
    constexpr std::size_t no_job = static_cast<std::size_t>(-1);
    std::vector<animation::FlightJob> jobs;
    std::vector<std::size_t> job_of(pending.size(), no_job);
    std::vector<std::optional<animation::MotionRecord>> wander_of(pending.size());
    animation::Motions& motions = scene.animation.motions;
    for (std::size_t i = 0; i < pending.size(); ++i) {
        const Pending& p = pending[i];
        const contracts::Transform& placed = scene.shapes.transforms[p.shape];
        const float body = placed.m[0][0];  // a sphere's radius (core/shapes/sphere.h)
        if (p.swarm != nullptr) {
            animation::FlightParams params = p.swarm->swarm.flight;
            params.seed = firefly_seed(params.seed, p.firefly);  // swarm.h, step 1
            job_of[i] = jobs.size();
            jobs.push_back({std::move(params), p.start, body});
            continue;
        }
        if (p.motion == nullptr) {
            continue;
        }
        const std::string what = p.what + "'s motion";
        const toml::table& t = r.table(*p.motion, what);
        const std::string_view kind = r.text(t, "kind", what);
        const contracts::Float3 center = contracts::translation(placed);
        if (kind == "wander") {
            r.only(t, {"kind", "reach", "speed", "seed"}, what);
            const float reach = read_positive(r, t, "reach", what);
            const float speed = read_positive(r, t, "speed", what);
            const std::uint64_t seed = read_seed(r, t, what);
            try {
                motions.wanders.push_back(animation::make_wander(center, reach, speed, seed, body, obstacles));
            } catch (const std::invalid_argument& refused) {
                r.fail(*p.motion, what + ": " + refused.what());
            }
            const animation::MotionRecord record{animation::MotionKind::wander,
                                                 static_cast<std::uint32_t>(motions.wanders.size() - 1)};
            const animation::Extent reach_box = animation::extent(motions, record);
            if (!inside_world(reach_box.min, reach_box.max, body)) {
                r.fail(*p.motion, what + " could carry " + p.what + " out of the world: it must stay " + world_words);
            }
            wander_of[i] = record;
        } else if (kind == "flight") {
            r.only(t, {"kind", "min", "max", "targets", "speed", "clearance", "circle", "swoop", "drift", "seed"},
                   what);
            animation::FlightParams params = read_flight_params(r, t, what, names, moving, scene);
            if (!inside_world(params.volume.min, params.volume.max, body)) {
                r.fail(t, what + "'s volume, grown by the sphere's radius, must lie " + world_words);
            }
            job_of[i] = jobs.size();
            jobs.push_back({std::move(params), center, body});
        } else {
            r.fail(r.required(t, "kind", what),
                   "unknown motion kind '" + std::string(kind) + "'; known kinds: wander, flight");
        }
    }

    std::vector<animation::Flight> flights;
    try {
        flights = animation::make_flights(jobs, obstacles);
    } catch (const animation::FlightsError& refused) {
        // The lowest job that failed, reported against its own line, without
        // the job number, which is no line of the file.
        const std::string message = refused.what();
        const std::string reason = message.substr(message.find(": ") + 2);
        for (std::size_t i = 0; i < pending.size(); ++i) {
            if (job_of[i] == refused.job) {
                const Pending& p = pending[i];
                if (p.swarm != nullptr) {
                    r.fail(*p.swarm->table, p.what + "'s flight: " + reason);
                }
                r.fail(*p.motion, p.what + "'s motion: " + reason);
            }
        }
        throw;  // unreachable: every job is some shape's
    }

    for (std::size_t i = 0; i < pending.size(); ++i) {
        const Pending& p = pending[i];
        std::optional<animation::MotionRecord> motion = wander_of[i];
        if (job_of[i] != no_job) {
            motions.flights.push_back(std::move(flights[job_of[i]]));
            motion = animation::MotionRecord{animation::MotionKind::flight,
                                             static_cast<std::uint32_t>(motions.flights.size() - 1)};
        }
        if (motion) {
            scene.animation.movers.push_back({p.shape, *motion});
        }
        const std::uint32_t light = scene.shape_lights[p.shape];
        if (p.swarm != nullptr) {
            // A flight glow with the swarm's flash and dim (swarm.h, step 3).
            animation::ScheduleGlow glow;
            glow.schedule = motions.flights[motion->index].flashes;
            glow.flash = p.swarm->swarm.flash;
            glow.dim = p.swarm->swarm.dim;
            scene.animation.glows.schedules.push_back(glow);
            const animation::GlowRecord record{animation::GlowKind::schedule,
                                               static_cast<std::uint32_t>(scene.animation.glows.schedules.size() - 1)};
            scene.animation.glowers.push_back({scene.lights[light].index, record});
        } else if (p.glow != nullptr) {
            if (light == lights::no_light) {
                r.fail(*p.glow, p.what + " has a glow, and is not a light");
            }
            const animation::GlowRecord glow = make_glow(r, p, motion ? &*motion : nullptr, scene);
            scene.animation.glowers.push_back({scene.lights[light].index, glow});
        }
    }
}

// The swarms (swarm.h): each read, its fireflies' starts drawn clear of the
// still shapes, and each firefly added as a sphere and a light, pending its
// flight and glow, after every shape before it.
void read_swarms(const Reader& r, const toml::array& all,
                 const std::map<std::string, std::uint32_t, std::less<>>& material_index,
                 const std::map<std::string, std::uint32_t, std::less<>>& names, std::vector<bool>& moving,
                 const contracts::Obstacles& obstacles, SceneDescription& scene, std::deque<SwarmEntry>& swarms,
                 std::vector<Pending>& pending) {
    std::size_t number = 0;
    for (const toml::node& node : all) {
        const std::string what = "swarm " + std::to_string(++number);
        const toml::table& t = r.table(node, what);
        r.only(t,
               {"count", "radius", "material", "min", "max", "targets", "speed", "clearance", "circle", "swoop",
                "drift", "flash", "dim", "seed"},
               what);
        SwarmEntry entry;
        entry.table = &t;
        entry.what = what;
        const toml::node& count_node = r.required(t, "count", what);
        const std::optional<std::int64_t> count = count_node.is_integer() ? count_node.value<std::int64_t>()
                                                                           : std::nullopt;
        if (!count || *count < 1 || *count > static_cast<std::int64_t>(max_swarm)) {
            r.fail(count_node, what + "'s count must be an integer from 1 to " + std::to_string(max_swarm));
        }
        entry.swarm.count = static_cast<std::uint32_t>(*count);
        entry.swarm.radius = read_positive(r, t, "radius", what);
        const std::string_view material = r.text(t, "material", what);
        const auto found = material_index.find(material);
        if (found == material_index.end()) {
            r.fail(r.required(t, "material", what),
                   what + " uses material '" + std::string(material) + "', which is not defined");
        }
        if (scene.materials[found->second].kind != materials::MaterialKind::emissive) {
            r.fail(r.required(t, "material", what), what + "'s material must be emissive: its fireflies are lights");
        }
        entry.swarm.flight = read_flight_params(r, t, what, names, moving, scene);
        if (!inside_world(entry.swarm.flight.volume.min, entry.swarm.flight.volume.max, entry.swarm.radius)) {
            r.fail(t, what + "'s volume, grown by its radius, must lie " + world_words);
        }
        entry.swarm.flash = read_positive(r, t, "flash", what);
        if (!(entry.swarm.flash < 1.0f)) {
            r.fail(r.required(t, "flash", what), what + "'s flash must be under a second");
        }
        entry.swarm.dim = r.number(t, "dim", what);
        if (!(entry.swarm.dim >= 0.0f && entry.swarm.dim < 1.0f)) {
            r.fail(r.required(t, "dim", what), what + "'s dim must be in [0, 1)");
        }
        swarms.push_back(std::move(entry));
        const SwarmEntry& swarm = swarms.back();
        for (std::uint32_t i = 0; i < swarm.swarm.count; ++i) {
            contracts::Float3 start;
            try {
                start = firefly_start(swarm.swarm, i, obstacles);  // swarm.h, step 2
            } catch (const std::invalid_argument& refused) {
                r.fail(t, what + ": " + refused.what());
            }
            const std::uint32_t index = add_sphere(scene, start, swarm.swarm.radius, found->second);
            moving.push_back(true);
            pending.push_back({index, what + "'s firefly " + std::to_string(i), nullptr, nullptr, &swarm, i, start});
        }
    }
}

void read_shapes(const Reader& r, const toml::array& all,
                 const std::map<std::string, std::uint32_t, std::less<>>& material_index, SceneDescription& scene,
                 std::vector<Pending>& pending, std::map<std::string, std::uint32_t, std::less<>>& names) {
    shapes::Shapes& shapes = scene.shapes;
    std::size_t number = 0;
    for (const toml::node& node : all) {
        const std::string what = "shape " + std::to_string(++number);
        const toml::table& t = r.table(node, what);
        const std::string_view kind = r.text(t, "kind", what);
        const auto index = static_cast<std::uint32_t>(shapes.records.size());
        if (const toml::node* name_node = t.get("name")) {
            const std::string_view name = r.text(*name_node, what + "'s name");
            if (!names.emplace(std::string(name), index).second) {
                r.fail(*name_node, "the name '" + std::string(name) + "' is used twice");
            }
        }

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
            r.only(t, {"kind", "name", "center", "radius", "material", "motion", "glow"}, what);
            const contracts::Float3 center = r.triple(t, "center", what);
            const float radius = r.number(t, "radius", what);
            if (!(radius > 0.0f)) {
                r.fail(r.required(t, "radius", what), what + "'s radius must be greater than 0");
            }
            if (!inside_world(center, center, radius)) {
                r.fail(t, what + " must lie " + world_words);
            }
            (void)add_sphere(scene, center, radius, material());
            if (t.contains("motion") || t.contains("glow")) {
                pending.push_back({index, what, t.get("motion"), t.get("glow")});
            }
        } else if (kind == "box") {
            r.only(t, {"kind", "name", "min", "max", "material"}, what);
            const contracts::Float3 min = r.triple(t, "min", what);
            const contracts::Float3 max = r.triple(t, "max", what);
            if (!below(min, max)) {
                r.fail(t, what + "'s min must be below its max on every axis");
            }
            if (!inside_world(min, max, 0.0)) {
                r.fail(t, what + " must lie " + world_words);
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

SceneDescription read_scene(const Reader& r, const toml::table& root) {
    r.only(root, {"camera", "environment", "textures", "materials", "shapes", "swarms"}, "the scene");
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
    std::vector<Pending> pending;
    std::map<std::string, std::uint32_t, std::less<>> names;
    read_shapes(r, *list, material_index, scene, pending, names);

    // The still shapes are the written ones that do not move; every swarm's
    // fireflies move, so none is added to them.
    std::vector<bool> moving(scene.shapes.records.size(), false);
    for (const Pending& p : pending) {
        moving[p.shape] = moving[p.shape] || p.motion != nullptr;
    }
    const StillShapes obstacles(scene.shapes, moving);
    std::deque<SwarmEntry> swarms;  // pending fireflies point into it, so it never moves an entry
    if (const toml::node* swarm_node = root.get("swarms")) {
        const toml::array* swarm_list = swarm_node->as_array();
        if (swarm_list == nullptr) {
            r.fail(*swarm_node, "'swarms' must be an array of tables, written [[swarms]]");
        }
        read_swarms(r, *swarm_list, material_index, names, moving, obstacles, scene, swarms, pending);
    }
    make_animation(r, pending, names, moving, obstacles, scene);
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
