#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/frame/schedule.h"

namespace serenity::scene {

// Axis: Scene content (reading it).
//
// A scene description: everything about a run that is data rather than
// code. Today that is the frame's schedule; the geometry, materials, lights,
// fireflies and camera join it as their kinds are written. Scene files live
// in scenes/, written by hand, in TOML, so they can carry comments while they
// are art-directed. Reading them is toml++'s, pinned by commit (NOTICE); this
// file's implementation is the only one that includes it.
//
// The format, today:
//
//   [frame]
//   passes = ["test_pattern"]     # pass names, in order; at least one
//
// Every key is checked. A missing [frame] or passes, a passes that is not a
// non-empty array of strings, an unknown pass name, or a key nobody reads is
// an Error, never a default or an ignored line: a typo in a scene file must
// fail the run that reads it, not render something else (E.2, E.14). Each
// Error names the file and the line, and an unknown pass name lists the
// known ones.
//
// Not performance-sensitive: read once, at start-up.

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

struct SceneDescription {
    frame::Schedule schedule;
};

// Reads the scene file at `path`.
SceneDescription load(const std::filesystem::path& path);

// Reads a scene from `text`, naming it `source` in errors. For tests, and
// for load() itself.
SceneDescription parse(std::string_view text, std::string_view source);

}  // namespace serenity::scene
