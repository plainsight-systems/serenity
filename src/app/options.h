#pragma once

#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>

namespace serenity::app {

// Axis: Presentation.
//
// What the window is asked to show, from its command line:
//
//   serenity --graph FILE [--scene FILE]
//
// The frame graph is required: there is no default and no path baked into
// the program, so what the window shows is always named where it is run.
// The scene is required only by a graph whose passes read one, and the
// renderer says so if it is missing (metal/frame/renderer.h). parse() throws
// OptionsError naming the argument for an unknown option, a missing value,
// or a missing --graph (E.2, E.14). It reads neither file; they are loaded
// after parsing succeeds (core/frame/graph_file.h, core/scene/scene.h).
class OptionsError : public std::runtime_error {
public:
    explicit OptionsError(const std::string& what) : std::runtime_error(what) {}
};

struct Options {
    std::filesystem::path graph;
    std::filesystem::path scene;  // empty when none was given
};

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

}  // namespace serenity::app
