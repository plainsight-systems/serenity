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
//   serenity --graph FILE
//
// The frame graph is required: there is no default and no path baked into
// the program, so what the window shows is always named where it is run.
// --scene joins it when scenes have content to read. parse() throws
// OptionsError naming the argument for an unknown option, a missing value,
// or a missing --graph (E.2, E.14). It does not read the file; the graph is
// loaded after parsing succeeds (core/frame/graph_file.h).
class OptionsError : public std::runtime_error {
public:
    explicit OptionsError(const std::string& what) : std::runtime_error(what) {}
};

struct Options {
    std::filesystem::path graph;
};

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

}  // namespace serenity::app
