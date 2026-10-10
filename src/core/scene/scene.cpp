#include "core/scene/scene.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <toml++/toml.hpp>

#include "core/animation/flight.h"
#include "core/animation/glow.h"
#include "core/animation/motion_error.h"
#include "core/animation/wander.h"
#include "core/camera/thin_lens.h"
#include "core/contracts/medium.h"
#include "core/contracts/obstacles.h"
#include "core/scene/swarm.h"

namespace serenity::scene {

namespace {

// Names in the file, to the index of what they name (T.42).
using NameIndex = std::map<std::string, std::uint32_t, std::less<>>;

// Reads one scene file; every error names the file and the line, in the
// form compilers use, so editors can jump to it. The text of an error is
// built only when it is raised.
class Reader {
public:
    explicit Reader(std::string_view source) : source_(source) {}

    [[noreturn]] void fail(const toml::source_region& where, const std::string& message) const {
        std::ostringstream s;
        s << source_ << ':' << where.begin.line << ": " << message;
        throw SceneError(s.str());
    }

    [[noreturn]] void fail(const toml::node& node, const std::string& message) const {
        fail(node.source(), message);
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw SceneError(std::string(source_) + ": " + message);
    }

    // Every key in `t` must be one of `allowed`.
    void only(const toml::table& t, std::initializer_list<std::string_view> allowed, std::string_view what) const {
        for (const auto& [key, value] : t) {
            if (std::ranges::none_of(allowed, [&](std::string_view name) { return key.str() == name; })) {
                fail(key.source(), "unknown key '" + std::string(key.str()) + "' in " + std::string(what));
            }
        }
    }

    const toml::node& required(const toml::table& t, std::string_view key, std::string_view what) const {
        const toml::node* node = t.get(key);
        if (!node) {
            fail(t, std::string(what) + " has no '" + std::string(key) + "'");
        }
        return *node;
    }

    const toml::table& table(const toml::node& node, std::string_view what) const {
        const toml::table* t = node.as_table();
        if (!t) {
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

    // An integer from `least` to `most`, or a SceneError saying so in `range`
    // words (ES.3: the one way the file's integers are read).
    std::int64_t integer(const toml::node& node, std::int64_t least, std::int64_t most,
                         const std::string& message) const {
        const std::optional<std::int64_t> value = node.is_integer() ? node.value<std::int64_t>() : std::nullopt;
        if (!value || *value < least || *value > most) {
            fail(node, message);
        }
        return *value;
    }

    contracts::Float3 triple(const toml::node& node, std::string_view what) const {
        const toml::array* a = node.as_array();
        if (!a || a->size() != 3) {
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

// A number as an error message writes it, the shortest decimal that reads
// back as it: 0.0001, 10 (ES.45: the message formats the constant it
// states, so the two cannot disagree).
std::string words(double value) {
    std::array<char, 64> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                            std::chars_format::fixed);
    return error == std::errc{} ? std::string(buffer.data(), end) : std::to_string(value);
}

std::string words(float value) {
    std::array<char, 64> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                            std::chars_format::fixed);
    return error == std::errc{} ? std::string(buffer.data(), end) : std::to_string(value);
}

// The flash-length messages say "under a second": the spacing a flight
// keeps its flashes apart by, which this ties them to.
static_assert(animation::flash_spacing == 1.0, "a flash's messages say 'under a second'");

// The wait messages say "an hour": most_wait, which this ties them to.
static_assert(animation::most_wait == 3600.0, "a wait's messages say 'an hour'");

// What the world's bound says in an error: "within 1000 km of the origin
// on every axis", from world_extent (I.22: built when an error is, not at
// static initialization).
std::string world_words() {
    return "within " + words(world_extent / 1000.0) + " km of the origin on every axis";
}

// Whether [lo, hi] lies within the world on every axis (scene.h), in double.
bool inside_world(double lo, double hi) {
    return std::isfinite(lo) && std::isfinite(hi) && lo >= -world_extent && hi <= world_extent;
}

bool inside_world(contracts::Float3 lo, contracts::Float3 hi, double grown) {
    for (int axis = 0; axis < 3; ++axis) {
        if (!inside_world(static_cast<double>(contracts::component(lo, axis)) - grown,
                          static_cast<double>(contracts::component(hi, axis)) + grown)) {
            return false;
        }
    }
    return true;
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
    if (const std::optional<std::string_view> reason = camera::invalid(camera)) {
        r.fail(t, "the camera cannot be framed: " + std::string(*reason));
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
    // A radiance, 0 or more in every channel: all 0 is a black sky.
    const auto color = [&](std::string_view key) {
        const contracts::Float3 c = r.triple(t, key, "[environment]");
        if (!within(c, 0.0f, std::numeric_limits<float>::max())) {
            r.fail(r.required(t, key, "[environment]"),
                   "[environment]'s " + std::string(key) + " must be 0 or more in every channel");
        }
        return c;
    };
    lights::GradientSkyData sky{};
    sky.zenith = color("zenith");
    sky.horizon = color("horizon");
    return sky;
}

std::uint64_t read_seed(const Reader& r, const toml::table& t, const std::string& what) {
    return static_cast<std::uint64_t>(r.integer(r.required(t, "seed", what), 0,
                                                std::numeric_limits<std::int64_t>::max(),
                                                what + "'s seed must be an integer, 0 or more"));
}

// A seed the shaders keep in 32 bits: a wood's or a swirl's.
std::uint32_t read_seed_32(const Reader& r, const toml::table& t, const std::string& what) {
    constexpr std::uint32_t most = std::numeric_limits<std::uint32_t>::max();
    const std::uint64_t seed = read_seed(r, t, what);
    if (seed > most) {
        r.fail(r.required(t, "seed", what), what + "'s seed must be at most " + std::to_string(most));
    }
    return static_cast<std::uint32_t>(seed);
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

// A wait in seconds at `key`: 0 or more, and at most most_wait
// (core/animation/flight.h), which bounds an opening's flashes.
double read_wait(const Reader& r, const toml::table& t, std::string_view key, const std::string& what) {
    const float wait = read_at_least_zero(r, t, key, what);
    if (!(static_cast<double>(wait) <= animation::most_wait)) {
        r.fail(r.required(t, key, what), what + "'s " + std::string(key) + " must be at most " +
                                             words(animation::most_wait) + " s, an hour");
    }
    return wait;
}

// A glow's wake (core/animation/glow.h), if it has one: `at` a finite
// number, `ramp` 0 or more. None is lit from the start.
std::optional<animation::Wake> read_wake(const Reader& r, const toml::table& t, const std::string& what) {
    const toml::node* node = t.get("wake");
    if (!node) {
        return std::nullopt;
    }
    const std::string wake_what = what + "'s wake";
    const toml::table& wake = r.table(*node, wake_what);
    r.only(wake, {"at", "ramp"}, wake_what);
    return animation::Wake{.at = r.number(wake, "at", wake_what),
                           .ramp = read_at_least_zero(r, wake, "ramp", wake_what)};
}

// A written flight's prelude (core/animation/flight.h), if it has one: hold,
// with `until`; perch, with `at` and `until`. None flies at once.
animation::Prelude read_prelude(const Reader& r, const toml::table& t, const std::string& what) {
    const toml::node* node = t.get("prelude");
    if (!node) {
        return animation::NoPrelude{};
    }
    const std::string prelude_what = what + "'s prelude";
    const toml::table& prelude = r.table(*node, prelude_what);
    const std::string_view kind = r.text(prelude, "kind", prelude_what);
    if (kind == "hold") {
        r.only(prelude, {"kind", "until"}, prelude_what);
        return animation::Hold{.until = read_wait(r, prelude, "until", prelude_what)};
    }
    if (kind == "perch") {
        r.only(prelude, {"kind", "at", "until"}, prelude_what);
        return animation::Perch{.at = r.triple(prelude, "at", prelude_what),
                                .until = read_wait(r, prelude, "until", prelude_what)};
    }
    r.fail(r.required(prelude, "kind", prelude_what),
           "unknown prelude kind '" + std::string(kind) + "'; known kinds: hold, perch");
}

// A swarm's start (swarm.h), if it has one: air; above, with `depth`; perch,
// with a box `min` to `max` inside the world and a `linger` of a least and
// a most. None is air.
SwarmStart read_start(const Reader& r, const toml::table& t, const std::string& what) {
    const toml::node* node = t.get("start");
    if (!node) {
        return AirStart{};
    }
    const std::string start_what = what + "'s start";
    const toml::table& start = r.table(*node, start_what);
    const std::string_view kind = r.text(start, "kind", start_what);
    if (kind == "air") {
        r.only(start, {"kind"}, start_what);
        return AirStart{};
    }
    if (kind == "above") {
        r.only(start, {"kind", "depth"}, start_what);
        return AboveStart{.depth = read_positive(r, start, "depth", start_what)};
    }
    if (kind == "perch") {
        r.only(start, {"kind", "min", "max", "linger"}, start_what);
        PerchStart perch;
        perch.box = {r.triple(start, "min", start_what), r.triple(start, "max", start_what)};
        if (!below(perch.box.min, perch.box.max)) {
            r.fail(start, start_what + "'s min must be below its max on every axis");
        }
        if (!inside_world(perch.box.min, perch.box.max, 0.0)) {
            r.fail(start, start_what + "'s box must lie " + world_words());
        }
        const toml::node& linger_node = r.required(start, "linger", start_what);
        const toml::array* linger = linger_node.as_array();
        if (!linger || linger->size() != 2) {
            r.fail(linger_node, start_what + "'s linger must be an array of two numbers, a least and a most");
        }
        perch.linger_least = r.number((*linger)[0], start_what + "'s linger");
        perch.linger_most = r.number((*linger)[1], start_what + "'s linger");
        if (!(perch.linger_least >= 0.0 && perch.linger_most >= perch.linger_least)) {
            r.fail(linger_node, start_what + "'s linger's least must be 0 or more, and its most at least its least");
        }
        return perch;
    }
    r.fail(r.required(start, "kind", start_what),
           "unknown start kind '" + std::string(kind) + "'; known kinds: air, above, perch");
}

// A swarm's wake (swarm.h), if it has one: `from` 0 or more, `to` from it
// to most_wait, `power` above 0, `ramp` 0 or more. None is lit from the
// start.
std::optional<SwarmWake> read_swarm_wake(const Reader& r, const toml::table& t, const std::string& what) {
    const toml::node* node = t.get("wake");
    if (!node) {
        return std::nullopt;
    }
    const std::string wake_what = what + "'s wake";
    const toml::table& wake = r.table(*node, wake_what);
    r.only(wake, {"from", "to", "power", "ramp"}, wake_what);
    SwarmWake w;
    w.from = read_at_least_zero(r, wake, "from", wake_what);
    w.to = read_wait(r, wake, "to", wake_what);
    if (!(w.to >= w.from)) {
        r.fail(r.required(wake, "to", wake_what), wake_what + "'s to must be at least its from");
    }
    w.power = read_positive(r, wake, "power", wake_what);
    w.ramp = read_at_least_zero(r, wake, "ramp", wake_what);
    return w;
}

// A color at `key`, within [0, 1] in every channel.
contracts::Float3 read_unit_color(const Reader& r, const toml::table& t, std::string_view key,
                                  const std::string& what) {
    const contracts::Float3 color = r.triple(t, key, what);
    if (!within(color, 0.0f, 1.0f)) {
        r.fail(r.required(t, key, what), what + "'s " + std::string(key) + " must be within [0, 1] in every channel");
    }
    return color;
}

textures::CheckerData read_checker(const Reader& r, const toml::table& t, const std::string& what) {
    r.only(t, {"kind", "size", "a", "b"}, what);
    textures::CheckerData checker{};
    checker.size = read_positive(r, t, "size", what);
    checker.a = r.triple(t, "a", what);
    checker.b = r.triple(t, "b", what);
    return checker;
}

textures::WoodData read_wood(const Reader& r, const toml::table& t, const std::string& what) {
    r.only(t, {"kind", "light", "dark", "ring", "board", "seed"}, what);
    textures::WoodData wood{};
    wood.light = read_unit_color(r, t, "light", what);
    wood.dark = read_unit_color(r, t, "dark", what);
    wood.ring = r.number(t, "ring", what);
    if (!(wood.ring >= textures::wood_least_ring)) {
        r.fail(r.required(t, "ring", what),
               what + "'s ring must be at least " + words(textures::wood_least_ring) + " m");
    }
    wood.board = r.number(t, "board", what);
    if (!(wood.board >= textures::wood_least_board && wood.board <= textures::wood_most_board)) {
        r.fail(r.required(t, "board", what), what + "'s board must be from " + words(textures::wood_least_board) +
                                                 " to " + words(textures::wood_most_board) + " m");
    }
    wood.seed = read_seed_32(r, t, what);
    return wood;
}

textures::SwirlData read_swirl(const Reader& r, const toml::table& t, const std::string& what) {
    r.only(t, {"kind", "a", "b", "vanes", "twist", "seed"}, what);
    textures::SwirlData swirl{};
    swirl.a = read_unit_color(r, t, "a", what);
    swirl.b = read_unit_color(r, t, "b", what);
    swirl.vanes = static_cast<std::uint32_t>(
        r.integer(r.required(t, "vanes", what), 1, textures::swirl_most_vanes,
                  what + "'s vanes must be an integer from 1 to " + std::to_string(textures::swirl_most_vanes)));
    swirl.twist = r.number(t, "twist", what);
    swirl.seed = read_seed_32(r, t, what);
    return swirl;
}

// Textures, in name order; returns each name's index into the records.
NameIndex read_textures(const Reader& r, const toml::table& all, SceneDescription& description) {
    NameIndex index;
    for (const auto& [key, node] : all) {
        const std::string name{key.str()};
        const std::string what = "texture '" + name + "'";
        const toml::table& t = r.table(node, what);
        const std::string_view kind = r.text(t, "kind", what);
        if (kind == "checker") {
            description.textures.push_back(
                {textures::TextureKind::checker, static_cast<std::uint32_t>(description.checkers.size())});
            description.checkers.push_back(read_checker(r, t, what));
        } else if (kind == "wood") {
            description.textures.push_back(
                {textures::TextureKind::wood, static_cast<std::uint32_t>(description.woods.size())});
            description.woods.push_back(read_wood(r, t, what));
        } else if (kind == "swirl") {
            description.textures.push_back(
                {textures::TextureKind::swirl, static_cast<std::uint32_t>(description.swirls.size())});
            description.swirls.push_back(read_swirl(r, t, what));
        } else {
            r.fail(r.required(t, "kind", what),
                   "unknown texture kind '" + std::string(kind) + "'; known kinds: checker, wood, swirl");
        }
        index.emplace(name, static_cast<std::uint32_t>(description.textures.size() - 1));
    }
    return index;
}

media::AbsorbingData read_absorbing(const Reader& r, const toml::table& t, const std::string& what) {
    r.only(t, {"kind", "tint", "tint_distance"}, what);
    const contracts::Float3 tint = r.triple(t, "tint", what);
    if (!(tint.x > 0.0f && tint.x <= 1.0f && tint.y > 0.0f && tint.y <= 1.0f && tint.z > 0.0f && tint.z <= 1.0f)) {
        r.fail(r.required(t, "tint", what), what + "'s tint must be within (0, 1] in every channel");
    }
    const float distance = read_positive(r, t, "tint_distance", what);
    // absorption = -ln(tint) / tint_distance (media/absorbing.h), in double,
    // 0 or more for any tint in (0, 1], and checked within float's range
    // before it is narrowed (ES.46).
    const auto absorption = [&](float channel) {
        const double a = -std::log(static_cast<double>(channel)) / static_cast<double>(distance);
        if (!(a <= static_cast<double>(std::numeric_limits<float>::max()))) {
            r.fail(r.required(t, "tint_distance", what),
                   what + "'s tint over its tint_distance absorbs past what a float holds");
        }
        return static_cast<float>(a);
    };
    media::AbsorbingData absorbing{};
    absorbing.absorption = {absorption(tint.x), absorption(tint.y), absorption(tint.z)};
    return absorbing;
}

// Media, in name order (contract 12); returns each name's index into the
// records.
NameIndex read_media(const Reader& r, const toml::table& all, SceneDescription& description) {
    NameIndex index;
    for (const auto& [key, node] : all) {
        const std::string name{key.str()};
        const std::string what = "medium '" + name + "'";
        const toml::table& t = r.table(node, what);
        const std::string_view kind = r.text(t, "kind", what);
        if (kind == "absorbing") {
            description.media.push_back(
                {media::MediumKind::absorbing, static_cast<std::uint32_t>(description.absorbings.size())});
            description.absorbings.push_back(read_absorbing(r, t, what));
        } else {
            r.fail(r.required(t, "kind", what),
                   "unknown medium kind '" + std::string(kind) + "'; known kinds: absorbing");
        }
        index.emplace(name, static_cast<std::uint32_t>(description.media.size() - 1));
    }
    return index;
}

// A rough or coated surface's base: a color, or a texture by name, exactly
// one.
struct Albedo {
    contracts::Float3 color{};
    contracts::TextureReference texture{contracts::no_texture};
};

Albedo read_albedo(const Reader& r, const toml::table& t, const std::string& what, const NameIndex& texture_index) {
    const bool has_color = t.contains("color");
    const bool has_texture = t.contains("texture");
    if (has_color == has_texture) {
        r.fail(t, what + " needs exactly one of 'color' and 'texture'");
    }
    Albedo albedo;
    if (has_color) {
        albedo.color = r.triple(t, "color", what);
        return albedo;
    }
    const std::string_view texture_name = r.text(t, "texture", what);
    const auto found = texture_index.find(texture_name);
    if (found == texture_index.end()) {
        r.fail(r.required(t, "texture", what),
               what + " uses texture '" + std::string(texture_name) + "', which is not defined");
    }
    albedo.texture.index = found->second;
    return albedo;
}

materials::CoatedData read_coated(const Reader& r, const toml::table& t, const std::string& what,
                                  const NameIndex& texture_index, const SceneDescription& description) {
    r.only(t, {"kind", "color", "texture", "ior"}, what);
    const Albedo base = read_albedo(r, t, what, texture_index);
    materials::CoatedData coated{};
    coated.color = base.color;
    coated.texture = base.texture;
    // Within [0, 1]: the base's bounces under the coat sum to
    // rho / (1 - rho F_in), finite only below 1 / F_in (coated.h).
    if (coated.texture.index == contracts::no_texture && !within(coated.color, 0.0f, 1.0f)) {
        r.fail(r.required(t, "color", what), what + "'s color must be within [0, 1]");
    }
    // A texture's colors too: wood's and swirl's are read within it; a
    // checker's are not otherwise checked.
    if (coated.texture.index != contracts::no_texture) {
        const textures::TextureRecord texture = description.textures[coated.texture.index];
        if (texture.kind == textures::TextureKind::checker &&
            (!within(description.checkers[texture.index].a, 0.0f, 1.0f) ||
             !within(description.checkers[texture.index].b, 0.0f, 1.0f))) {
            r.fail(r.required(t, "texture", what), what + "'s texture's colors must be within [0, 1]");
        }
    }
    coated.ior = r.number(t, "ior", what);
    if (!(coated.ior > 1.0f)) {
        r.fail(r.required(t, "ior", what), what + "'s ior must be greater than 1");
    }
    // Once a material, at load (F.8): the shaders read it.
    coated.escape = static_cast<float>(materials::internal_escape(coated.ior));
    return coated;
}

materials::DielectricData read_dielectric(const Reader& r, const toml::table& t, const std::string& what) {
    r.only(t, {"kind", "ior"}, what);
    materials::DielectricData dielectric{};
    dielectric.ior = r.number(t, "ior", what);
    if (!(dielectric.ior > 1.0f)) {
        r.fail(r.required(t, "ior", what), what + "'s ior must be greater than 1");
    }
    return dielectric;
}

materials::ConductorData read_conductor(const Reader& r, const toml::table& t, const std::string& what) {
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
    return conductor;
}

materials::EmissiveData read_emissive(const Reader& r, const toml::table& t, const std::string& what) {
    r.only(t, {"kind", "radiance"}, what);
    materials::EmissiveData emissive{};
    emissive.radiance = r.triple(t, "radiance", what);
    if (!within(emissive.radiance, 0.0f, std::numeric_limits<float>::max())) {
        r.fail(r.required(t, "radiance", what), what + "'s radiance must not be negative");
    }
    return emissive;
}

// Materials, in name order; returns each name's index into the records.
NameIndex read_materials(const Reader& r, const toml::table& all, const NameIndex& texture_index,
                         SceneDescription& description) {
    NameIndex index;
    const auto record = [&](materials::MaterialKind kind, std::size_t at) {
        description.materials.push_back({kind, static_cast<std::uint32_t>(at)});
    };
    for (const auto& [key, node] : all) {
        const std::string name{key.str()};
        const std::string what = "material '" + name + "'";
        const toml::table& t = r.table(node, what);
        const std::string_view kind = r.text(t, "kind", what);
        if (kind == "rough") {
            r.only(t, {"kind", "color", "texture"}, what);
            const Albedo base = read_albedo(r, t, what, texture_index);
            materials::RoughData rough{};
            rough.color = base.color;
            rough.texture = base.texture;
            record(materials::MaterialKind::rough, description.roughs.size());
            description.roughs.push_back(rough);
        } else if (kind == "coated") {
            const materials::CoatedData coated = read_coated(r, t, what, texture_index, description);
            record(materials::MaterialKind::coated, description.coateds.size());
            description.coateds.push_back(coated);
        } else if (kind == "dielectric") {
            record(materials::MaterialKind::dielectric, description.dielectrics.size());
            description.dielectrics.push_back(read_dielectric(r, t, what));
        } else if (kind == "conductor") {
            record(materials::MaterialKind::conductor, description.conductors.size());
            description.conductors.push_back(read_conductor(r, t, what));
        } else if (kind == "emissive") {
            record(materials::MaterialKind::emissive, description.emissives.size());
            description.emissives.push_back(read_emissive(r, t, what));
        } else {
            r.fail(r.required(t, "kind", what), "unknown material kind '" + std::string(kind) +
                                                    "'; known kinds: rough, coated, dielectric, conductor, emissive");
        }
        index.emplace(name, static_cast<std::uint32_t>(description.materials.size() - 1));
    }
    return index;
}

// A swarm as read (swarm.h), and where it was read, for errors: the table
// is the parsed file's, which outlives the reading (I.12: a reference that
// cannot be null, and copyable, C.12).
struct SwarmEntry {
    Swarm swarm;
    std::reference_wrapper<const toml::table> table;
    std::string what;  // "swarm N"
};

// A Pending that is not a swarm's firefly.
constexpr std::size_t no_swarm = std::numeric_limits<std::size_t>::max();

// A shape's motion and glow, as read: made once every shape is read, when
// the still shapes it must keep clear of and the names it circles are known.
// A written shape's are its tables; a swarm's firefly's, its swarm's.
struct Pending {
    std::uint32_t shape = 0;
    std::string what;                    // "shape N", or "swarm N's firefly i", for errors
    const toml::node* motion = nullptr;  // a written shape's motion's table, if any
    const toml::node* glow = nullptr;    // and its glow's
    std::size_t swarm = no_swarm;        // a swarm's firefly: its swarm's index
    std::uint32_t firefly = 0;           // which of it
    contracts::Float3 start{};           // and where it starts (swarm.h, step 2)
    animation::Prelude prelude;          // what it does before its loop (swarm.h, step 4)
    std::optional<animation::Wake> wake;  // and when its glow wakes (step 3)
};

// The indices of the shapes that do not move.
std::vector<std::uint32_t> still_of(const std::vector<bool>& moving) {
    std::vector<std::uint32_t> still;
    for (std::uint32_t i = 0; i < moving.size(); ++i) {
        if (!moving[i]) {
            still.push_back(i);
        }
    }
    return still;
}

// The still shapes, as contract 11 asks of them: each shape kind's exact
// distance and touch tests over every shape that does not move, placed
// once (shapes::StillShapes, core/shapes/shapes.h). Holds its own copy of
// what the tests read, so the shapes added after it is made (a swarm's
// fireflies, which move) change nothing it answers. Not copyable, as no
// Obstacles is (C.67).
class StillObstacles final : public contracts::Obstacles {
public:
    explicit StillObstacles(const shapes::Shapes& all, const std::vector<bool>& moving)
        : still_{all, still_of(moving)} {}

    double distance(contracts::Float3 point) const override { return still_.distance(point); }

    bool touches(const contracts::Box& box) const override { return still_.touches(shapes::Bounds{box.min, box.max}); }

private:
    shapes::StillShapes still_;
};

// Reads the shapes, the swarms and every motion and glow into one scene
// description: what each step reads and writes is this class's state, not
// arguments threaded through every function (I.23, F.20).
class ShapeReading {
public:
    ShapeReading(const Reader& r, SceneDescription& description, const NameIndex& material_index,
                 const NameIndex& medium_index)
        : r_(r), description_(description), material_index_(material_index), medium_index_(medium_index) {}

    // [[shapes]]: each read, a sphere or a box; motions and glows pending.
    void read_shapes(const toml::array& all);

    // The still shapes are the written ones that do not move; every swarm's
    // fireflies move, so none is added to them.
    std::vector<bool> moving() const;

    // [[swarms]]: each read, its fireflies' starts drawn clear of
    // `obstacles`, each firefly added as a sphere and a light, pending its
    // flight and glow, after every shape before it.
    void read_swarms(const toml::array& all, const contracts::Obstacles& obstacles);

    // Every motion and glow, made now that every shape is read: motions
    // against the still shapes, glows from their motions. Wanders are made
    // as they come; every flight, written or a swarm's, is gathered and made
    // at once, in parallel (core/animation/flight.h, make_flights), then
    // each motion and glow is recorded in the order the shapes come.
    void make_animation(const contracts::Obstacles& obstacles);

private:
    std::uint32_t material(const toml::table& t, const std::string& what) const;
    std::uint32_t interior(const toml::table& t, const std::string& what, std::uint32_t worn) const;
    std::uint32_t add_sphere(contracts::Float3 center, float radius, std::uint32_t worn_index);
    void read_sphere(const toml::table& t, const std::string& what, std::uint32_t index);
    void read_box(const toml::table& t, const std::string& what);
    animation::FlightParams read_flight_params(const toml::table& t, const std::string& what) const;
    std::optional<animation::MotionRecord> make_wander(const Pending& p, const toml::table& t, const std::string& what,
                                                       float body, const contracts::Obstacles& obstacles);
    animation::GlowRecord make_glow(const Pending& p, const animation::MotionRecord* motion);
    [[noreturn]] void report(const animation::FlightsError& refused, const std::vector<std::size_t>& job_of) const;
    const toml::node& flight_node(const Pending& p) const;

    const Reader& r_;
    SceneDescription& description_;
    const NameIndex& material_index_;
    const NameIndex& medium_index_;
    NameIndex names_;  // the shapes', what a flight circles
    std::vector<Pending> pending_;
    std::vector<bool> moving_;  // per shape, once the shapes are read
    std::vector<SwarmEntry> swarms_;  // pending fireflies name theirs by index
};

std::uint32_t ShapeReading::material(const toml::table& t, const std::string& what) const {
    const std::string_view name = r_.text(t, "material", what);
    const auto found = material_index_.find(name);
    if (found == material_index_.end()) {
        r_.fail(r_.required(t, "material", what),
                what + " uses material '" + std::string(name) + "', which is not defined");
    }
    return found->second;
}

// The medium inside the shape (contract 12), named by `interior`, or air
// without one: only a shape light passes into, one wearing a dielectric, may
// be filled.
std::uint32_t ShapeReading::interior(const toml::table& t, const std::string& what, std::uint32_t worn) const {
    const toml::node* node = t.get("interior");
    if (!node) {
        return contracts::no_medium;
    }
    const std::string_view name = r_.text(*node, what + "'s interior");
    const auto found = medium_index_.find(name);
    if (found == medium_index_.end()) {
        r_.fail(*node, what + "'s interior is medium '" + std::string(name) + "', which is not defined");
    }
    if (description_.materials[worn].kind != materials::MaterialKind::dielectric) {
        r_.fail(*node, what + " has an interior, and light cannot pass into it: its material is not a dielectric");
    }
    return found->second;
}

// A sphere of `radius` at `center` wearing material `worn_index`, and, if
// that is emissive, a sphere light; its index.
std::uint32_t ShapeReading::add_sphere(contracts::Float3 center, float radius, std::uint32_t worn_index) {
    SceneDescription& d = description_;
    const auto index = static_cast<std::uint32_t>(d.shapes.records.size());
    const materials::MaterialRecord worn = d.materials[worn_index];
    const bool glows = worn.kind == materials::MaterialKind::emissive;
    d.shape_lights.push_back(glows ? static_cast<std::uint32_t>(d.lights.size()) : lights::no_light);
    if (glows) {
        d.lights.push_back({lights::LightKind::sphere, static_cast<std::uint32_t>(d.sphere_lights.size())});
        d.sphere_lights.push_back({d.emissives[worn.index].radiance, index});
    }
    d.shapes.records.push_back({shapes::ShapeKind::sphere, 0u, worn_index, contracts::no_medium});
    d.shapes.transforms.push_back(contracts::placed(center, radius));
    return index;
}

void ShapeReading::read_sphere(const toml::table& t, const std::string& what, std::uint32_t index) {
    r_.only(t, {"kind", "name", "center", "radius", "material", "interior", "motion", "glow"}, what);
    const contracts::Float3 center = r_.triple(t, "center", what);
    const float radius = read_positive(r_, t, "radius", what);
    if (!inside_world(center, center, radius)) {
        r_.fail(t, what + " must lie " + world_words());
    }
    const std::uint32_t worn = material(t, what);
    const std::uint32_t added = add_sphere(center, radius, worn);
    description_.shapes.records[added].interior = interior(t, what, worn);
    if (t.contains("motion") || t.contains("glow")) {
        pending_.push_back({.shape = index, .what = what, .motion = t.get("motion"), .glow = t.get("glow")});
    }
}

void ShapeReading::read_box(const toml::table& t, const std::string& what) {
    r_.only(t, {"kind", "name", "min", "max", "material", "interior"}, what);
    const contracts::Float3 min = r_.triple(t, "min", what);
    const contracts::Float3 max = r_.triple(t, "max", what);
    if (!below(min, max)) {
        r_.fail(t, what + "'s min must be below its max on every axis");
    }
    if (!inside_world(min, max, 0.0)) {
        r_.fail(t, what + " must lie " + world_words());
    }
    const std::uint32_t worn = material(t, what);
    if (description_.materials[worn].kind == materials::MaterialKind::emissive) {
        r_.fail(r_.required(t, "material", what), what + " is a box; only a sphere may be emissive");
    }
    // Its corners as the file gives them, placed by the identity transform
    // (core/shapes/box.h).
    shapes::Shapes& all = description_.shapes;
    description_.shape_lights.push_back(lights::no_light);
    all.records.push_back(
        {shapes::ShapeKind::box, static_cast<std::uint32_t>(all.boxes.size()), worn, interior(t, what, worn)});
    all.transforms.push_back(contracts::placed({0.0f, 0.0f, 0.0f}, 1.0f));
    all.boxes.push_back(shapes::BoxData{min, 0u, max, 0u});
}

void ShapeReading::read_shapes(const toml::array& all) {
    std::size_t number = 0;
    for (const toml::node& node : all) {
        const std::string what = "shape " + std::to_string(++number);
        const toml::table& t = r_.table(node, what);
        const std::string_view kind = r_.text(t, "kind", what);
        const auto index = static_cast<std::uint32_t>(description_.shapes.records.size());
        if (const toml::node* name_node = t.get("name")) {
            const std::string_view name = r_.text(*name_node, what + "'s name");
            if (!names_.emplace(std::string(name), index).second) {
                r_.fail(*name_node, "the name '" + std::string(name) + "' is used twice");
            }
        }
        if (kind == "sphere") {
            read_sphere(t, what, index);
        } else if (kind == "box") {
            read_box(t, what);
        } else {
            r_.fail(r_.required(t, "kind", what),
                    "unknown shape kind '" + std::string(kind) + "'; known kinds: sphere, box");
        }
    }
    if (description_.shapes.records.empty()) {
        r_.fail("the scene has no shapes");
    }
    moving_.assign(description_.shapes.records.size(), false);
    for (const Pending& p : pending_) {
        moving_[p.shape] = moving_[p.shape] || p.motion;
    }
}

std::vector<bool> ShapeReading::moving() const {
    return moving_;
}

// A flight's numbers from table `t`, read as a written flight's motion and
// a swarm both write them (scene.h); not yet made.
animation::FlightParams ShapeReading::read_flight_params(const toml::table& t, const std::string& what) const {
    animation::FlightParams params;
    params.volume = {r_.triple(t, "min", what), r_.triple(t, "max", what)};
    if (!below(params.volume.min, params.volume.max)) {
        r_.fail(t, what + "'s min must be below its max on every axis");
    }
    const toml::node& list_node = r_.required(t, "targets", what);
    const toml::array* list = list_node.as_array();
    if (!list) {
        r_.fail(list_node, what + "'s targets must be an array of shape names");
    }
    for (const toml::node& name_node : *list) {
        const std::string_view name = r_.text(name_node, what + "'s target");
        const auto found = names_.find(name);
        if (found == names_.end()) {
            r_.fail(name_node, what + " circles '" + std::string(name) + "', which is not defined");
        }
        const std::uint32_t target = found->second;
        if (description_.shapes.records[target].kind != shapes::ShapeKind::sphere || moving_[target]) {
            r_.fail(name_node, what + " circles '" + std::string(name) + "', which is not a still sphere");
        }
        const contracts::Transform& at = description_.shapes.transforms[target];
        params.targets.push_back({contracts::translation(at), contracts::scale(at)});
    }
    params.speed = read_positive(r_, t, "speed", what);
    params.clearance = read_at_least_zero(r_, t, "clearance", what);
    params.weights = {read_at_least_zero(r_, t, "circle", what), read_at_least_zero(r_, t, "swoop", what),
                      read_at_least_zero(r_, t, "drift", what)};
    if (params.weights[0] + params.weights[1] + params.weights[2] <= 0.0f) {
        r_.fail(t, what + "'s circle, swoop and drift are all 0");
    }
    if (params.weights[0] > 0.0f && params.targets.empty()) {
        r_.fail(r_.required(t, "circle", what), what + " circles, and names no targets");
    }
    params.seed = read_seed(r_, t, what);
    return params;
}

void ShapeReading::read_swarms(const toml::array& all, const contracts::Obstacles& obstacles) {
    std::size_t number = 0;
    for (const toml::node& node : all) {
        const std::string what = "swarm " + std::to_string(++number);
        const toml::table& t = r_.table(node, what);
        r_.only(t,
                {"count", "radius", "material", "min", "max", "targets", "speed", "clearance", "circle", "swoop",
                 "drift", "flash", "dim", "seed", "start", "wake"},
                what);
        SwarmEntry entry{.swarm = {}, .table = t, .what = what};
        entry.swarm.count = static_cast<std::uint32_t>(
            r_.integer(r_.required(t, "count", what), 1, max_swarm,
                       what + "'s count must be an integer from 1 to " + std::to_string(max_swarm)));
        entry.swarm.radius = read_positive(r_, t, "radius", what);
        const std::uint32_t worn = material(t, what);
        if (description_.materials[worn].kind != materials::MaterialKind::emissive) {
            r_.fail(r_.required(t, "material", what), what + "'s material must be emissive: its fireflies are lights");
        }
        entry.swarm.flight = read_flight_params(t, what);
        if (!inside_world(entry.swarm.flight.volume.min, entry.swarm.flight.volume.max, entry.swarm.radius)) {
            r_.fail(t, what + "'s volume, grown by its radius, must lie " + world_words());
        }
        entry.swarm.flash = read_positive(r_, t, "flash", what);
        if (!(entry.swarm.flash < animation::flash_spacing)) {
            r_.fail(r_.required(t, "flash", what), what + "'s flash must be under a second");
        }
        entry.swarm.dim = r_.number(t, "dim", what);
        if (!(entry.swarm.dim >= 0.0f && entry.swarm.dim < 1.0f)) {
            r_.fail(r_.required(t, "dim", what), what + "'s dim must be in [0, 1)");
        }
        entry.swarm.start = read_start(r_, t, what);
        entry.swarm.wake = read_swarm_wake(r_, t, what);
        // A perched firefly waits its wake and its linger: at most most_wait
        // in all (core/animation/flight.h, step P1). Compared as make_firefly
        // compares it, against what most_wait leaves after the wake's to
        // (swarm.h), so the two cannot disagree at a rounding.
        if (const PerchStart* perch = std::get_if<PerchStart>(&entry.swarm.start)) {
            const double latest = entry.swarm.wake ? entry.swarm.wake->to : 0.0;
            if (!(perch->linger_most <= animation::most_wait - latest)) {
                r_.fail(entry.swarm.wake ? *t.get("wake") : *t.get("start"),
                        what + "'s wake's to plus its linger's most must be at most " + words(animation::most_wait) +
                            " s, an hour");
            }
        }
        swarms_.push_back(std::move(entry));
        const std::size_t swarm_index = swarms_.size() - 1;
        const Swarm& swarm = swarms_.back().swarm;
        for (std::uint32_t i = 0; i < swarm.count; ++i) {
            const Firefly firefly = [&] {
                try {
                    return make_firefly(swarm, i, obstacles);  // swarm.h, steps 2 to 3
                } catch (const animation::MotionError& refused) {
                    r_.fail(t, what + ": " + refused.what());
                }
            }();
            const std::uint32_t index = add_sphere(firefly.start, swarm.radius, worn);
            moving_.push_back(true);
            pending_.push_back({.shape = index,
                                .what = what + "'s firefly " + std::to_string(i),
                                .swarm = swarm_index,
                                .firefly = i,
                                .start = firefly.start,
                                .prelude = firefly.prelude,
                                .wake = firefly.wake});
        }
    }
}

std::optional<animation::MotionRecord> ShapeReading::make_wander(const Pending& p, const toml::table& t,
                                                                 const std::string& what, float body,
                                                                 const contracts::Obstacles& obstacles) {
    r_.only(t, {"kind", "reach", "speed", "seed"}, what);
    const contracts::Float3 anchor = contracts::translation(description_.shapes.transforms[p.shape]);
    const animation::WanderParams params{.anchor = anchor,
                                         .reach = read_positive(r_, t, "reach", what),
                                         .speed = read_positive(r_, t, "speed", what),
                                         .seed = read_seed(r_, t, what)};
    // Within the world before it is made: so every number make_wander takes
    // is in its range, and only a refusal is left for it to report.
    if (!inside_world(anchor, anchor, static_cast<double>(params.reach) + body)) {
        r_.fail(*p.motion, what + " could carry " + p.what + " out of the world: it must stay " + world_words());
    }
    animation::Motions& motions = description_.animation.motions;
    try {
        motions.wanders.push_back(animation::make_wander(params, body, obstacles));
    } catch (const animation::MotionError& refused) {
        r_.fail(*p.motion, what + ": " + refused.what());
    }
    const animation::MotionRecord record{animation::MotionKind::wander,
                                         static_cast<std::uint32_t>(motions.wanders.size() - 1)};
    const animation::Extent reach_box = animation::extent(motions, record);
    if (!inside_world(reach_box.min, reach_box.max, body)) {
        r_.fail(*p.motion, what + " could carry " + p.what + " out of the world: it must stay " + world_words());
    }
    return record;
}

animation::GlowRecord ShapeReading::make_glow(const Pending& p, const animation::MotionRecord* motion) {
    const std::string what = p.what + "'s glow";
    const toml::table& t = r_.table(*p.glow, what);
    const std::string_view kind = r_.text(t, "kind", what);
    const auto dim_of = [&]() {
        const float dim = r_.number(t, "dim", what);
        if (!(dim >= 0.0f && dim < 1.0f)) {
            r_.fail(r_.required(t, "dim", what), what + "'s dim must be in [0, 1)");
        }
        return dim;
    };
    animation::Glows& glows = description_.animation.glows;
    if (kind == "rhythm") {
        r_.only(t, {"kind", "period", "flash", "dim", "seed", "wake"}, what);
        animation::Rhythm rhythm;
        rhythm.period = read_positive(r_, t, "period", what);
        if (!(rhythm.period >= animation::least_period)) {
            r_.fail(r_.required(t, "period", what),
                    what + "'s period must be at least " + words(animation::least_period) + " s");
        }
        rhythm.flash = read_positive(r_, t, "flash", what);
        if (rhythm.flash > rhythm.period / 2.0) {
            r_.fail(r_.required(t, "flash", what), what + "'s flash must be at most half its period");
        }
        rhythm.dim = dim_of();
        rhythm.seed = read_seed(r_, t, what);
        rhythm.wake = read_wake(r_, t, what);
        glows.rhythms.push_back(rhythm);
        return {animation::GlowKind::rhythm, static_cast<std::uint32_t>(glows.rhythms.size() - 1)};
    }
    if (kind == "flight") {
        r_.only(t, {"kind", "flash", "dim", "wake"}, what);
        if (!motion || motion->kind != animation::MotionKind::flight) {
            r_.fail(*p.glow, what + " follows its flight, and " + p.what + " does not fly");
        }
        animation::ScheduleGlow glow;
        glow.schedule = description_.animation.motions.flights[motion->index].flashes;
        glow.flash = read_positive(r_, t, "flash", what);
        if (!(glow.flash < animation::flash_spacing)) {
            r_.fail(r_.required(t, "flash", what), what + "'s flash must be under a second");
        }
        glow.dim = dim_of();
        glow.wake = read_wake(r_, t, what);
        glows.schedules.push_back(std::move(glow));
        return {animation::GlowKind::schedule, static_cast<std::uint32_t>(glows.schedules.size() - 1)};
    }
    r_.fail(r_.required(t, "kind", what), "unknown glow kind '" + std::string(kind) + "'; known kinds: rhythm, flight");
}

// The lowest job that failed, reported against its own line, without the
// job number, which is no line of the file.
void ShapeReading::report(const animation::FlightsError& refused, const std::vector<std::size_t>& job_of) const {
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        if (job_of[i] == refused.job) {
            const Pending& p = pending_[i];
            r_.fail(flight_node(p), p.what + (p.swarm != no_swarm ? "'s flight: " : "'s motion: ") + refused.reason);
        }
    }
    throw std::logic_error("a flight was refused that no shape asked for");
}

// Where a flight's errors are reported: a swarm's firefly's at its swarm's
// table, a written firefly's at its motion (ES.3: report() and the reach's
// check both ask).
const toml::node& ShapeReading::flight_node(const Pending& p) const {
    if (p.swarm != no_swarm) {
        return swarms_[p.swarm].table.get();
    }
    return *p.motion;
}

void ShapeReading::make_animation(const contracts::Obstacles& obstacles) {
    constexpr std::size_t no_job = std::numeric_limits<std::size_t>::max();
    std::vector<animation::FlightJob> jobs;
    std::vector<std::size_t> job_of(pending_.size(), no_job);
    std::vector<std::optional<animation::MotionRecord>> wander_of(pending_.size());
    animation::Motions& motions = description_.animation.motions;
    for (std::size_t i = 0; i < pending_.size(); ++i) {
        const Pending& p = pending_[i];
        const contracts::Transform& placed = description_.shapes.transforms[p.shape];
        const float body = contracts::scale(placed);  // a sphere's radius (core/shapes/sphere.h)
        if (p.swarm != no_swarm) {
            animation::FlightParams params = swarms_[p.swarm].swarm.flight;
            params.seed = firefly_seed(params.seed, p.firefly);  // swarm.h, step 1
            job_of[i] = jobs.size();
            jobs.push_back({.params = std::move(params), .start = p.start, .body = body, .prelude = p.prelude});
            continue;
        }
        if (!p.motion) {
            continue;
        }
        const std::string what = p.what + "'s motion";
        const toml::table& t = r_.table(*p.motion, what);
        const std::string_view kind = r_.text(t, "kind", what);
        if (kind == "wander") {
            wander_of[i] = make_wander(p, t, what, body, obstacles);
        } else if (kind == "flight") {
            r_.only(t,
                    {"kind", "min", "max", "targets", "speed", "clearance", "circle", "swoop", "drift", "seed",
                     "prelude"},
                    what);
            animation::FlightParams params = read_flight_params(t, what);
            if (!inside_world(params.volume.min, params.volume.max, body)) {
                r_.fail(t, what + "'s volume, grown by the sphere's radius, must lie " + world_words());
            }
            job_of[i] = jobs.size();
            jobs.push_back({.params = std::move(params),
                            .start = contracts::translation(placed),
                            .body = body,
                            .prelude = read_prelude(r_, t, what)});
        } else {
            r_.fail(r_.required(t, "kind", what),
                    "unknown motion kind '" + std::string(kind) + "'; known kinds: wander, flight");
        }
    }

    std::vector<animation::Flight> flights;
    try {
        flights = animation::make_flights(jobs, obstacles, animation::flight_workers());
    } catch (const animation::FlightsError& refused) {
        report(refused, job_of);
    }

    for (std::size_t i = 0; i < pending_.size(); ++i) {
        const Pending& p = pending_[i];
        std::optional<animation::MotionRecord> motion = wander_of[i];
        if (job_of[i] != no_job) {
            // Its reach, the volume grown to hold its prelude, grown by its
            // body, must lie in the world (scene.h).
            const animation::Extent reach = animation::extent(flights[job_of[i]]);
            if (!inside_world(reach.min, reach.max, jobs[job_of[i]].body)) {
                r_.fail(flight_node(p),
                        p.what + "'s flight could carry it out of the world: it must stay " + world_words());
            }
            motions.flights.push_back(std::move(flights[job_of[i]]));
            motion = animation::MotionRecord{animation::MotionKind::flight,
                                             static_cast<std::uint32_t>(motions.flights.size() - 1)};
        }
        if (motion) {
            description_.animation.movers.push_back({p.shape, *motion});
        }
        const std::uint32_t light = description_.shape_lights[p.shape];
        if (p.swarm != no_swarm) {
            // A flight glow with the swarm's flash and dim, and the
            // firefly's wake (swarm.h, step 4).
            animation::ScheduleGlow glow;
            glow.schedule = motions.flights[motion->index].flashes;
            glow.flash = swarms_[p.swarm].swarm.flash;
            glow.dim = swarms_[p.swarm].swarm.dim;
            glow.wake = p.wake;
            animation::Glows& glows = description_.animation.glows;
            glows.schedules.push_back(std::move(glow));
            const animation::GlowRecord record{animation::GlowKind::schedule,
                                               static_cast<std::uint32_t>(glows.schedules.size() - 1)};
            description_.animation.glowers.push_back({description_.lights[light].index, record});
        } else if (p.glow) {
            if (light == lights::no_light) {
                r_.fail(*p.glow, p.what + " has a glow, and is not a light");
            }
            const animation::GlowRecord glow = make_glow(p, motion ? &*motion : nullptr);
            description_.animation.glowers.push_back({description_.lights[light].index, glow});
        }
    }
}

// A table of the root's at `key`, or an empty one if the file has none.
const toml::table& optional_table(const Reader& r, const toml::table& root, std::string_view key,
                                  const std::string& what, const toml::table& none) {
    const toml::node* node = root.get(key);
    return node ? r.table(*node, what) : none;
}

SceneDescription read_scene(const Reader& r, const toml::table& root) {
    r.only(root, {"camera", "environment", "textures", "materials", "media", "shapes", "swarms"}, "the scene");
    SceneDescription description{};

    description.camera = read_camera(r, r.table(r.required(root, "camera", "the scene"), "[camera]"));
    description.environment =
        read_environment(r, r.table(r.required(root, "environment", "the scene"), "[environment]"));

    const toml::table no_entries;
    const NameIndex texture_index =
        read_textures(r, optional_table(r, root, "textures", "[textures]", no_entries), description);
    const NameIndex material_index =
        read_materials(r, optional_table(r, root, "materials", "[materials]", no_entries), texture_index, description);
    const NameIndex medium_index = read_media(r, optional_table(r, root, "media", "[media]", no_entries), description);

    const toml::node& shape_node = r.required(root, "shapes", "the scene");
    const toml::array* shape_list = shape_node.as_array();
    if (!shape_list) {
        r.fail(shape_node, "'shapes' must be an array of tables, written [[shapes]]");
    }
    ShapeReading reading{r, description, material_index, medium_index};
    reading.read_shapes(*shape_list);

    const StillObstacles obstacles{description.shapes, reading.moving()};
    if (const toml::node* swarm_node = root.get("swarms")) {
        const toml::array* swarm_list = swarm_node->as_array();
        if (!swarm_list) {
            r.fail(*swarm_node, "'swarms' must be an array of tables, written [[swarms]]");
        }
        reading.read_swarms(*swarm_list, obstacles);
    }
    reading.make_animation(obstacles);
    description.light_counts.lights = static_cast<std::uint32_t>(description.lights.size());
    description.light_counts.spheres = static_cast<std::uint32_t>(description.sphere_lights.size());
    return description;
}

}  // namespace

SceneDescription parse(std::string_view text, std::string_view source) {
    toml::table root;
    try {
        root = toml::parse(text, source);
    } catch (const toml::parse_error& error) {
        Reader{source}.fail(error.source(), std::string(error.description()));
    }
    return read_scene(Reader{source}, root);
}

SceneDescription load(const std::filesystem::path& path) {
    // Sized from the file, and read whole or refused: a read that fails, a
    // directory, or a file that ends early is a SceneError, never an empty or a
    // shorter scene (I.10, SL.io.2).
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    std::ifstream file{path, std::ios::binary};
    if (error || !file || size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw SceneError("cannot read scene file " + path.string());
    }
    std::string text(static_cast<std::size_t>(size), '\0');
    file.read(text.data(), static_cast<std::streamsize>(size));
    if (!file || file.gcount() != static_cast<std::streamsize>(size)) {
        throw SceneError("cannot read scene file " + path.string());
    }
    return parse(text, path.string());
}

}  // namespace serenity::scene
