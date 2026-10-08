#include "core/scene/scene.h"

#include <fstream>
#include <sstream>

#include <toml++/toml.hpp>

namespace serenity::scene {

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
    for (frame::PassKind kind : frame::all_pass_kinds()) {
        if (!names.empty()) {
            names += ", ";
        }
        names += frame::name(kind);
    }
    return names;
}

// Every key in `table` must be one of `allowed`.
void only(std::string_view source, const toml::table& table, std::initializer_list<std::string_view> allowed,
          std::string_view where) {
    for (const auto& [key, value] : table) {
        bool known = false;
        for (std::string_view name : allowed) {
            known = known || key.str() == name;
        }
        if (!known) {
            throw Error(at(source, key.source(),
                           "unknown key '" + std::string(key.str()) + "' in " + std::string(where)));
        }
    }
}

frame::Schedule read_schedule(std::string_view source, const toml::table& root) {
    const toml::node* frame_node = root.get("frame");
    if (frame_node == nullptr) {
        throw Error(std::string(source) + ": missing [frame]");
    }
    const toml::table* frame_table = frame_node->as_table();
    if (frame_table == nullptr) {
        throw Error(at(source, *frame_node, "'frame' must be a table"));
    }
    only(source, *frame_table, {"passes"}, "[frame]");

    const toml::node* passes_node = frame_table->get("passes");
    if (passes_node == nullptr) {
        throw Error(at(source, *frame_node, "[frame] has no 'passes'"));
    }
    const toml::array* passes = passes_node->as_array();
    if (passes == nullptr || passes->empty()) {
        throw Error(at(source, *passes_node, "'passes' must be a non-empty array of pass names"));
    }

    frame::Schedule schedule;
    schedule.passes.reserve(passes->size());
    for (const toml::node& entry : *passes) {
        const std::optional<std::string_view> text = entry.value<std::string_view>();
        if (!text) {
            throw Error(at(source, entry, "each pass must be a name, in quotes"));
        }
        const std::optional<frame::PassKind> kind = frame::pass_kind(*text);
        if (!kind) {
            throw Error(at(source, entry,
                           "unknown pass '" + std::string(*text) + "'; known passes: " + known_pass_names()));
        }
        schedule.passes.push_back(*kind);
    }
    return schedule;
}

}  // namespace

SceneDescription parse(std::string_view text, std::string_view source) {
    toml::table root;
    try {
        root = toml::parse(text, source);
    } catch (const toml::parse_error& error) {
        throw Error(at(source, error.source(), std::string(error.description())));
    }
    only(source, root, {"frame"}, "the scene");

    SceneDescription scene;
    scene.schedule = read_schedule(source, root);
    return scene;
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
