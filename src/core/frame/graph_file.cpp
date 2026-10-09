#include "core/frame/graph_file.h"

#include <fstream>
#include <sstream>

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
        bool known = false;
        for (std::string_view name : allowed) {
            known = known || key.str() == name;
        }
        if (!known) {
            throw GraphFileError(at(source, key.source(),
                           "unknown key '" + std::string(key.str()) + "' in " + std::string(where)));
        }
    }
}

Schedule read_schedule(std::string_view source, const toml::table& root) {
    only(source, root, {"passes"}, "the frame graph");

    const toml::node* passes_node = root.get("passes");
    if (passes_node == nullptr) {
        throw GraphFileError(std::string(source) + ": no 'passes'");
    }
    const toml::array* passes = passes_node->as_array();
    if (passes == nullptr || passes->empty()) {
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
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw GraphFileError("cannot read frame graph file " + path.string());
    }
    std::ostringstream text;
    text << file.rdbuf();
    if (!file && !file.eof()) {
        throw GraphFileError("cannot read frame graph file " + path.string());
    }
    return parse_schedule(text.str(), path.string());
}

}  // namespace serenity::frame
