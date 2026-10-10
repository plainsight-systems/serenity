#include "core/frame/graph_file.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>

#include <toml++/toml.hpp>

namespace serenity::frame {

namespace {

// "<source>:<line>: <message>", the form compilers use, so editors can jump
// to it.
std::string at(std::string_view source, const toml::source_region& where, const std::string& message) {
    std::ostringstream s;
    s << source << ':' << where.begin.line << ": " << message;
    return s.str();
}

std::string at(std::string_view source, const toml::node& node, const std::string& message) {
    return at(source, node.source(), message);
}

std::string known_pass_names() {
    std::string names;
    for (PassKind kind : all_pass_kinds()) {
        if (!names.empty()) {
            names += ", ";
        }
        names += name(kind);
    }
    return names;
}

// Every key in `table` must be one of `allowed`.
void only(std::string_view source, const toml::table& table, std::initializer_list<std::string_view> allowed,
          std::string_view where) {
    for (const auto& [key, value] : table) {
        if (std::ranges::none_of(allowed, [&](std::string_view allowed_key) { return key.str() == allowed_key; })) {
            throw GraphFileError(
                at(source, key.source(), "unknown key '" + std::string(key.str()) + "' in " + std::string(where)));
        }
    }
}

// The node in `table` at `key`, which must be there.
const toml::node& required(std::string_view source, const toml::table& table, std::string_view key,
                           std::string_view where) {
    const toml::node* node = table.get(key);
    if (!node) {
        throw GraphFileError(at(source, table, std::string(where) + " has no '" + std::string(key) + "'"));
    }
    return *node;
}

// `node` as a float: a number, finite and within float's range, checked
// while it is still a double, since narrowing one past that range is
// undefined (ES.46).
float finite_float(std::string_view source, const toml::node& node, std::string_view key) {
    const std::optional<double> value = node.value<double>();
    if (!value) {
        throw GraphFileError(at(source, node, "'" + std::string(key) + "' must be a number"));
    }
    if (!std::isfinite(*value) || std::abs(*value) > std::numeric_limits<float>::max()) {
        throw GraphFileError(at(source, node, "'" + std::string(key) + "' must be a finite number"));
    }
    return static_cast<float>(*value);
}

// The tone map's settings, each checked by the core's rule (schedule.h)
// and refused at its own line.
passes::ToneMap read_tone_map(std::string_view source, const toml::table& table) {
    only(source, table, {"exposure", "bloom"}, "[tone_map]");
    const toml::node& exposure = required(source, table, "exposure", "[tone_map]");
    const toml::node& bloom = required(source, table, "bloom", "[tone_map]");
    passes::ToneMap settings{};
    settings.exposure = finite_float(source, exposure, "exposure");
    settings.bloom = finite_float(source, bloom, "bloom");
    // Each alone, beside a value of the other the rule accepts.
    if (const std::optional<std::string> reason = invalid(passes::ToneMap{settings.exposure, 0.0f, {}})) {
        throw GraphFileError(at(source, exposure, *reason));
    }
    if (const std::optional<std::string> reason = invalid(passes::ToneMap{0.0f, settings.bloom, {}})) {
        throw GraphFileError(at(source, bloom, *reason));
    }
    return settings;
}

Schedule read_schedule(std::string_view source, const toml::table& root) {
    only(source, root, {"passes", "tone_map"}, "the frame graph");

    const toml::node* passes_node = root.get("passes");
    if (!passes_node) {
        throw GraphFileError(std::string(source) + ": no 'passes'");
    }
    const toml::array* passes = passes_node->as_array();
    if (!passes || passes->empty()) {
        throw GraphFileError(at(source, *passes_node, "'passes' must be a non-empty array of pass names"));
    }

    Schedule schedule;
    schedule.passes.reserve(passes->size());
    for (const toml::node& entry : *passes) {
        const std::optional<std::string_view> text = entry.value<std::string_view>();
        if (!text) {
            throw GraphFileError(at(source, entry, "each pass must be a name, in quotes"));
        }
        const std::optional<PassKind> kind = pass_kind(*text);
        if (!kind) {
            throw GraphFileError(at(source, entry,
                                    "unknown pass '" + std::string(*text) + "'; known passes: " + known_pass_names()));
        }
        schedule.passes.push_back(*kind);
    }

    // The tone map's settings, each mistake reported at the line it is
    // about; whether they come with their pass, by the rule for the whole.
    const toml::node* tone_map_node = root.get("tone_map");
    if (tone_map_node) {
        const toml::table* table = tone_map_node->as_table();
        if (!table) {
            throw GraphFileError(at(source, *tone_map_node, "'tone_map' must be a table, [tone_map]"));
        }
        schedule.tone_map = read_tone_map(source, *table);
    }
    if (const std::optional<std::string> reason = invalid(schedule)) {
        throw GraphFileError(at(source, *passes_node, *reason));
    }
    return schedule;
}

}  // namespace

Schedule parse_schedule(std::string_view text, std::string_view source) {
    toml::table root;
    try {
        root = toml::parse(text, source);
    } catch (const toml::parse_error& error) {
        throw GraphFileError(at(source, error.source(), std::string(error.description())));
    }
    return read_schedule(source, root);
}

Schedule load_schedule(const std::filesystem::path& path) {
    // Sized from the file, and read whole or refused: a read that fails, a
    // directory, or a file that ends early is an Error, never an empty or a
    // shorter graph (I.10, SL.io.2).
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    std::ifstream file{path, std::ios::binary};
    if (error || !file || size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw GraphFileError("cannot read frame graph file " + path.string());
    }
    std::string text(static_cast<std::size_t>(size), '\0');
    file.read(text.data(), static_cast<std::streamsize>(size));
    if (!file || file.gcount() != static_cast<std::streamsize>(size)) {
        throw GraphFileError("cannot read frame graph file " + path.string());
    }
    return parse_schedule(text, path.string());
}

}  // namespace serenity::frame
