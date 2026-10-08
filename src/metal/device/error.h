#pragma once

#include <stdexcept>
#include <string>

namespace serenity::metal {

// Axis: the Metal API surface (named in docs/architecture/change-axes.md once
// written).
//
// What the Metal backend throws when it cannot do what it was asked: no
// device, a device without ray tracing, a library Metal rejects, a function
// the library lacks. One type for the backend, so a caller can tell a Metal
// failure from any other (E.14); the message names the cause and, where Metal
// gave one, Metal's own description.
//
// The backend throws only while acquiring and building: a device, a library,
// a pipeline. Nothing on a frame's path throws.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace serenity::metal
