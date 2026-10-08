#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/frame/schedule.h"

namespace serenity::frame {

// Axis: Frame graph (reading it).
//
// A frame graph file: the passes a frame runs, in order, as data. Kept apart
// from scene files (change-axes.md: what is rendered is Scene content; which
// passes run, and in what order, is the Frame graph), so one scene runs under
// any graph: the naive estimator and ReSTIR side by side on the same scene,
// or the test pattern on none. Graph files live in graphs/, written by hand,
// in TOML. Reading them is toml++'s, pinned by commit (NOTICE).
//
// The format, today:
//
//   passes = ["test_pattern"]     # pass names, in order; at least one
//
// Every key is checked. A missing passes, a passes that is not a non-empty
// array of strings, an unknown pass name, or a key nobody reads is an Error,
// never a default or an ignored line: a typo must fail the run that reads
// it, not render something else (E.2, E.14). Each Error names the file and
// the line, and an unknown pass name lists the known ones.
//
// Not performance-sensitive: read once, at start-up.

class GraphFileError : public std::runtime_error {
public:
    explicit GraphFileError(const std::string& what) : std::runtime_error(what) {}
};

// Reads the frame graph file at `path`.
Schedule load_schedule(const std::filesystem::path& path);

// Reads a frame graph from `text`, naming it `source` in errors. For tests,
// and for load_schedule() itself.
Schedule parse_schedule(std::string_view text, std::string_view source);

}  // namespace serenity::frame
