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
// The format:
//
//   passes = ["path", "tone_map"] # pass names, in order; at least one
//
//   [tone_map]                    # exactly when a pass is tone_map
//   exposure = 0.0                # stops, within [-10, 10] (core/passes/tone_map.h)
//   bloom = 0.04                  # the fraction of light spread into glare
//
// Every key is checked. A missing passes, a passes that is not a non-empty
// array of strings, an unknown pass name, a [tone_map] table without a
// tone_map pass or the pass without the table, a setting missing or out of
// range, a schedule the core does not accept (schedule.h, invalid()), or a
// key nobody reads is a GraphFileError, never a default or an ignored line: a typo
// must fail the run that reads it, not render something else (E.2, E.14).
// Each GraphFileError names the file and the line, and an unknown pass name lists the
// known ones. A file that cannot be read whole, a directory's path or one
// that ends early, is a GraphFileError naming it, never an empty graph.
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
