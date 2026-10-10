#pragma once

#include <filesystem>
#include <span>
#include <stdexcept>
#include <variant>
#include <vector>

namespace serenity::measure {

// Axis: Measurement (the command line).
//
// serenity-measure: the reference made, and images judged against it, on
// the CPU, from Portable Float Maps the headless renderer wrote
// (headless/options.h, --format pfm; core/output/pfm.h). No GPU: the
// reference's images are the path graph's, rendered already, and what is
// left is arithmetic in double (core/measurement/). Two commands:
//
//   serenity-measure reference --out FILE BATCH...
//
//     Folds the batches, in the order given, into a reference
//     (core/measurement/reference.h) and writes it to FILE, a PFM; then
//     prints, a line each, `batches B`, `mse_floor X` and
//     `relative_mse_floor Y` (%.9g), the reference's own error. At least
//     two batches, so there is a floor. FILE must not exist: a reference is
//     never replaced by a run that might be another's (as the headless
//     renderer's directory must be empty, headless/options.h).
//
//   serenity-measure error --reference FILE IMAGE...
//
//     Prints a CSV: the header `image,mse,relative_mse`, then a row an
//     image, in the order given, its error against FILE
//     (core/measurement/error.h), each number %.9g; the image as its path
//     was given, quoted as RFC 4180 has it if it holds a comma, a double
//     quote or a line break, so no path splits its row.
//
// parse() checks the command line before any file is read and throws
// OptionsError naming the argument (E.2, E.14): an unknown command or
// option, a missing value, --out or --reference given twice or not at all,
// fewer than two batches, no images. Every file it names is then read and
// refused as core/output/pfm.h and core/measurement/ say; any failure ends
// the run with its message and a non-zero status, and nothing is printed as
// a result before the last file is judged, so a run that fails never leaves
// a partial table that reads like a whole one (E.2).
//
// The Makefile's `reference` and `convergence` targets run it: the
// reference's batches each from frames of their own, the images judged
// against it from frames none of the reference's use, so their noise is
// independent (core/measurement/error.h).

class OptionsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct MakeReference {
    std::filesystem::path out;
    std::vector<std::filesystem::path> batches;  // two or more
};

struct MeasureError {
    std::filesystem::path reference;
    std::vector<std::filesystem::path> images;  // one or more
};

// One of the two commands, each with only its own arguments, a
// std::variant (C.181, C.182) as elsewhere (core/animation/flight.h).
using Command = std::variant<MakeReference, MeasureError>;

// `args` are the arguments after the program's name.
Command parse(std::span<const char* const> args);

}  // namespace serenity::measure
