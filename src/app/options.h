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
//   serenity --scene FILE
//
// The scene is required: there is no default scene and no path baked into
// the program, so what the window shows is always named where it is run.
// parse() throws Error naming the argument for an unknown option, a missing
// value, or a missing --scene (E.2, E.14). It does not read the file; the
// scene is loaded after parsing succeeds (core/scene/scene.h).
class OptionsError : public std::runtime_error {
public:
    explicit OptionsError(const std::string& what) : std::runtime_error(what) {}
};

struct Options {
    std::filesystem::path scene;
};

// `args` are the arguments after the program's name.
Options parse(std::span<const char* const> args);

}  // namespace serenity::app
