#include "core/frame/graph_file.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
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

// `node` as a count: a TOML integer that fits 32 bits without a sign, so
// it narrows exactly (ES.46); its range is the core's rule's, checked
// after.
std::uint32_t count(std::string_view source, const toml::node& node, std::string_view key) {
    const std::optional<std::int64_t> value = node.is_integer() ? node.value<std::int64_t>() : std::nullopt;
    if (!value) {
        throw GraphFileError(at(source, node, "'" + std::string(key) + "' must be an integer"));
    }
    if (*value < 0 || *value > std::int64_t{std::numeric_limits<std::uint32_t>::max()}) {
        throw GraphFileError(at(source, node, "'" + std::string(key) + "', " + std::to_string(*value) +
                                                  ", is no count a setting takes "
                                                  "(core/contracts/reuse_settings.h, path_reuse_settings.h)"));
    }
    return static_cast<std::uint32_t>(*value);
}

// `node` as a switch: a TOML boolean.
bool switched_on(std::string_view source, const toml::node& node, std::string_view key) {
    const std::optional<bool> value = node.is_boolean() ? node.value<bool>() : std::nullopt;
    if (!value) {
        throw GraphFileError(at(source, node, "'" + std::string(key) + "' must be true or false"));
    }
    return *value;
}

// `node` as bias_correction: one of its two strings.
contracts::BiasCorrection bias_correction(std::string_view source, const toml::node& node) {
    const std::optional<std::string_view> value = node.value<std::string_view>();
    if (value == "basic") {
        return contracts::BiasCorrection::basic;
    }
    if (value == "raytraced") {
        return contracts::BiasCorrection::raytraced;
    }
    throw GraphFileError(at(source, node, "'bias_correction' must be \"basic\" or \"raytraced\""));
}

// Every [restir_di] key (graph_file.h), in the order the settings lay them
// out.
constexpr std::array<std::string_view, 12> restir_keys = {
    "candidates",       "history",          "permutation",      "neighbors",      "radius",
    "depth_threshold",  "normal_threshold", "bias_correction",  "visibility_reuse", "visibility_age",
    "visibility_distance", "bsdf_candidates"};

// Which [restir_di] keys a graph reads (graph_file.h): by its passes, and
// by two of the settings, bias_correction and visibility_reuse, which are
// read first for it.
struct RestirReads {
    bool temporal = false;
    bool spatial = false;
    bool merges = false;
    bool basic = false;   // a merge in basic mode: visibility_reuse read
    bool reused = false;  // and it on: visibility_age and visibility_distance read

    bool reads(std::string_view key) const {
        if (key == "candidates" || key == "bsdf_candidates") {
            return true;
        }
        if (key == "history" || key == "permutation") {
            return temporal;
        }
        if (key == "neighbors" || key == "radius") {
            return spatial;
        }
        if (key == "depth_threshold" || key == "normal_threshold" || key == "bias_correction") {
            return merges;
        }
        if (key == "visibility_reuse") {
            return basic;
        }
        if (key == "visibility_age" || key == "visibility_distance") {
            return reused;
        }
        return false;  // no key of [restir_di]'s
    }
};

// The keys `settings`' two deciding values and the pass `kinds` say are
// read; those two values, read first, into `settings`.
RestirReads restir_reads(std::string_view source, const toml::table& table, std::span<const PassKind> kinds,
                         passes::RestirDi& settings) {
    const auto has = [&](PassKind kind) { return std::ranges::find(kinds, kind) != kinds.end(); };
    RestirReads reads;
    reads.temporal = has(PassKind::restir_temporal);
    reads.spatial = has(PassKind::restir_spatial);
    reads.merges = reads.temporal || reads.spatial;
    if (reads.merges) {
        settings.bias_correction = bias_correction(source, required(source, table, "bias_correction", "[restir_di]"));
    }
    reads.basic = reads.merges && settings.bias_correction == contracts::BiasCorrection::basic;
    if (reads.basic) {
        const bool on =
            switched_on(source, required(source, table, "visibility_reuse", "[restir_di]"), "visibility_reuse");
        settings.visibility_reuse = on ? contracts::VisibilityReuse::on : contracts::VisibilityReuse::off;
    }
    reads.reused = settings.visibility_reuse == contracts::VisibilityReuse::on;
    return reads;
}

// Refuses, at its line, the first key in the file that `reads` does not
// read, unknown or not read by this graph: toml++ keeps a table's keys
// sorted, not in the file's order.
void refuse_unread(std::string_view source, const toml::table& table, const RestirReads& reads) {
    const toml::key* stray = nullptr;
    for (const auto& [key, value] : table) {
        if (!reads.reads(key.str()) && (stray == nullptr || key.source().begin.line < stray->source().begin.line)) {
            stray = &key;
        }
    }
    if (stray == nullptr) {
        return;
    }
    const std::string key(stray->str());
    throw GraphFileError(at(source, stray->source(),
                            std::ranges::find(restir_keys, key) == restir_keys.end()
                                ? "unknown key '" + key + "' in [restir_di]"
                                : "'" + key +
                                      "' is not read by this frame graph's passes "
                                      "(core/contracts/reuse_settings.h says which read it): a key nobody reads"));
}

// The settings each value is checked beside (read_restir_di): `settings`'
// two deciding values, the values the graphs are written with for every
// other key `reads` reads, zeros for the rest.
passes::RestirDi restir_baseline(passes::RestirDi settings, const RestirReads& reads) {
    settings.candidates = passes::written_candidates;
    settings.bsdf_candidates = passes::written_bsdf_candidates;
    if (reads.temporal) {
        settings.history = passes::written_history;
        settings.permutation = passes::written_permutation;
    }
    if (reads.spatial) {
        settings.neighbors = passes::written_neighbors;
        settings.radius = passes::written_radius;
    }
    if (reads.merges) {
        settings.depth_threshold = passes::written_depth_threshold;
        settings.normal_threshold = passes::written_normal_threshold;
    }
    if (reads.reused) {
        settings.visibility_age = passes::written_visibility_age;
        settings.visibility_distance = passes::written_visibility_distance;
    }
    return settings;
}

// ReSTIR DI's settings, for a graph of the pass `kinds`. Which keys are
// read depends on the passes and on two of the settings (graph_file.h), so
// those two are read first; then every key the graph reads must be there,
// and every key there must be read. Each value is checked by the core's one
// rule (schedule.h, invalid) at its own line: set alone into settings the
// rule accepts (restir_baseline), so the rule can refuse only it.
passes::RestirDi read_restir_di(std::string_view source, const toml::table& table, std::span<const PassKind> kinds) {
    passes::RestirDi settings{};  // every setting not read stays its zero
    const RestirReads reads = restir_reads(source, table, kinds, settings);
    refuse_unread(source, table, reads);
    const passes::RestirDi baseline = restir_baseline(settings, reads);

    // Reads `key` into `member` by `read`, and checks it alone.
    const auto take = [&]<typename T>(std::string_view key, T passes::RestirDi::*member, auto read) {
        const toml::node& node = required(source, table, key, "[restir_di]");
        settings.*member = read(node);
        passes::RestirDi alone = baseline;
        alone.*member = settings.*member;
        if (const std::optional<std::string> reason = invalid(alone, kinds)) {
            throw GraphFileError(at(source, node, *reason));
        }
    };
    const auto as_count = [&](std::string_view key) {
        return [&source, key](const toml::node& node) { return count(source, node, key); };
    };
    const auto as_float = [&](std::string_view key) {
        return [&source, key](const toml::node& node) { return finite_float(source, node, key); };
    };
    take("candidates", &passes::RestirDi::candidates, as_count("candidates"));
    take("bsdf_candidates", &passes::RestirDi::bsdf_candidates, as_count("bsdf_candidates"));
    if (reads.temporal) {
        take("history", &passes::RestirDi::history, as_count("history"));
        take("permutation", &passes::RestirDi::permutation, [&](const toml::node& node) {
            return switched_on(source, node, "permutation") ? contracts::Permutation::on : contracts::Permutation::off;
        });
    }
    if (reads.spatial) {
        take("neighbors", &passes::RestirDi::neighbors, as_count("neighbors"));
        take("radius", &passes::RestirDi::radius, as_float("radius"));
    }
    if (reads.merges) {
        take("depth_threshold", &passes::RestirDi::depth_threshold, as_float("depth_threshold"));
        take("normal_threshold", &passes::RestirDi::normal_threshold, as_float("normal_threshold"));
    }
    if (reads.reused) {
        take("visibility_age", &passes::RestirDi::visibility_age, as_count("visibility_age"));
        take("visibility_distance", &passes::RestirDi::visibility_distance, as_float("visibility_distance"));
    }
    return settings;
}

// ReSTIR PT's settings (core/contracts/path_reuse_settings.h): the four
// every graph with its passes reads, neighbors and sigma with
// restir_pt_spatial alone, confidence_cap with restir_pt_temporal, and
// cap_floor and cap_exponent with restir_pt_duplication. Each key the passes read must be written, each
// they do not must not be; each value is checked against its range alone,
// beside the written defaults (core/passes/restir_pt.h), so a mistake is
// reported at its own line.
passes::RestirPt read_restir_pt(std::string_view source, const toml::table& table, std::span<const PassKind> kinds) {
    const auto has = [&](PassKind kind) { return std::ranges::find(kinds, kind) != kinds.end(); };
    const bool spatial = has(PassKind::restir_pt_spatial);
    const bool temporal = has(PassKind::restir_pt_temporal);
    const bool duplication = has(PassKind::restir_pt_duplication);
    constexpr std::array<std::string_view, 9> keys = {"nee_candidates", "max_surfaces",   "footprint",
                                                      "roughness",      "neighbors",      "sigma",
                                                      "confidence_cap", "cap_floor",      "cap_exponent"};
    // The pass a key is read only with; none for the four every graph reads.
    const auto reader = [](std::string_view key) -> std::optional<PassKind> {
        if (key == "neighbors" || key == "sigma") {
            return PassKind::restir_pt_spatial;
        }
        if (key == "confidence_cap") {
            return PassKind::restir_pt_temporal;
        }
        if (key == "cap_floor" || key == "cap_exponent") {
            return PassKind::restir_pt_duplication;
        }
        return std::nullopt;
    };
    for (const auto& [key, value] : table) {
        if (std::ranges::find(keys, key.str()) == keys.end()) {
            throw GraphFileError(at(source, key.source(), "unknown key '" + std::string(key.str()) + "' in [restir_pt]"));
        }
        if (const std::optional<PassKind> pass = reader(key.str()); pass && !has(*pass)) {
            throw GraphFileError(at(source, key.source(),
                                    "'" + std::string(key.str()) + "' is read only with " + std::string(name(*pass)) +
                                        " (core/contracts/path_reuse_settings.h): a key nobody reads"));
        }
    }
    passes::RestirPt baseline{};
    baseline.nee_candidates = passes::written_nee_candidates;
    baseline.max_surfaces = passes::written_max_surfaces;
    baseline.footprint = passes::written_footprint;
    baseline.roughness = passes::written_roughness;
    if (spatial) {
        baseline.neighbors = passes::written_path_neighbors;
        baseline.sigma = passes::written_sigma;
    }
    if (temporal) {
        baseline.confidence_cap = passes::written_confidence_cap;
    }
    if (duplication) {
        baseline.cap_floor = passes::written_cap_floor;
        baseline.cap_exponent = passes::written_cap_exponent;
    }
    passes::RestirPt settings = baseline;
    const auto take = [&]<typename T>(std::string_view key, T passes::RestirPt::*member, auto read) {
        const toml::node& node = required(source, table, key, "[restir_pt]");
        settings.*member = read(node);
        passes::RestirPt alone = baseline;
        alone.*member = settings.*member;
        if (const std::optional<std::string> reason = invalid(alone, kinds)) {
            throw GraphFileError(at(source, node, *reason));
        }
    };
    const auto as_count = [&](std::string_view key) {
        return [&source, key](const toml::node& node) { return count(source, node, key); };
    };
    const auto as_float = [&](std::string_view key) {
        return [&source, key](const toml::node& node) { return finite_float(source, node, key); };
    };
    take("nee_candidates", &passes::RestirPt::nee_candidates, as_count("nee_candidates"));
    take("max_surfaces", &passes::RestirPt::max_surfaces, as_count("max_surfaces"));
    take("footprint", &passes::RestirPt::footprint, as_float("footprint"));
    take("roughness", &passes::RestirPt::roughness, as_float("roughness"));
    if (spatial) {
        take("neighbors", &passes::RestirPt::neighbors, as_count("neighbors"));
        take("sigma", &passes::RestirPt::sigma, as_float("sigma"));
    }
    if (temporal) {
        take("confidence_cap", &passes::RestirPt::confidence_cap, as_float("confidence_cap"));
        baseline.confidence_cap = settings.confidence_cap;  // the floor's bound is the cap written
    }
    if (duplication) {
        take("cap_floor", &passes::RestirPt::cap_floor, as_float("cap_floor"));
        take("cap_exponent", &passes::RestirPt::cap_exponent, as_float("cap_exponent"));
    }
    return settings;
}

Schedule read_schedule(std::string_view source, const toml::table& root) {
    only(source, root, {"passes", "tone_map", "restir_di", "restir_pt", "mnee"}, "the frame graph");

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
    // ReSTIR DI's, likewise; a table with no ReSTIR pass to read it is
    // refused at the table, before its keys, which no pass reads.
    if (const toml::node* restir_node = root.get("restir_di")) {
        const toml::table* table = restir_node->as_table();
        if (!table) {
            throw GraphFileError(at(source, *restir_node, "'restir_di' must be a table, [restir_di]"));
        }
        if (std::ranges::none_of(schedule.passes, [](PassKind kind) { return is_restir(kind); })) {
            throw GraphFileError(at(source, *restir_node, "the frame graph has [restir_di] settings and no ReSTIR DI pass"));
        }
        schedule.restir_di = read_restir_di(source, *table, schedule.passes);
    }
    if (const toml::node* restir_pt_node = root.get("restir_pt")) {
        const toml::table* table = restir_pt_node->as_table();
        if (!table) {
            throw GraphFileError(at(source, *restir_pt_node, "'restir_pt' must be a table, [restir_pt]"));
        }
        if (std::ranges::none_of(schedule.passes, [](PassKind kind) { return is_restir_pt(kind); })) {
            throw GraphFileError(
                at(source, *restir_pt_node, "the frame graph has [restir_pt] settings and no ReSTIR PT pass"));
        }
        schedule.restir_pt = read_restir_pt(source, *table, schedule.passes);
    }
    // Caustics by MNEE: true or false, read by the path pass and ReSTIR
    // PT's; a graph with neither refused at the key.
    if (const toml::node* mnee_node = root.get("mnee")) {
        const std::optional<bool> mnee = mnee_node->value_exact<bool>();
        if (!mnee) {
            throw GraphFileError(at(source, *mnee_node, "'mnee' must be true or false"));
        }
        schedule.mnee = *mnee;
        if (schedule.mnee && std::ranges::none_of(schedule.passes, [](PassKind kind) {
                return kind == PassKind::path || is_restir_pt(kind);
            })) {
            throw GraphFileError(at(source, *mnee_node, "the frame graph sets mnee and has no path or ReSTIR PT pass "
                                                        "to read it"));
        }
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
    // directory, or a file that ends early is a GraphFileError, never an empty or a
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
