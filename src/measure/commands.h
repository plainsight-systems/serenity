#pragma once

#include <string>
#include <string_view>

#include "measure/options.h"

namespace serenity::measure {

// Axis: Measurement (the commands).
//
// What serenity-measure's two commands do (measure/options.h states each),
// as functions of the command parse() made: the files read, the reference
// written, and what the command prints, returned whole, so main() prints
// it once, after the last file is judged (E.2: a run that fails prints no
// partial result). Kept apart from main() so the tests call them directly
// (tests/measure_test.cpp), the program's own run reduced to printing the
// result or the message.
//
// Not performance-sensitive: a handful of images, read once.

// The reference command: the batches folded in the order given and the
// reference written to its --out, by exclusive create; returns `batches B`,
// `mse_floor X` and `relative_mse_floor Y`, a line each (%.9g). Throws what
// reading a batch, folding it in or creating --out throws (core/output/
// pfm.h, core/measurement/reference.h).
std::string run(const MakeReference& command);

// The error command: returns the CSV, `image,mse,relative_mse` then a row
// an image in the order given, its path as csv_field() writes it. Throws
// what reading an image or judging it throws (core/measurement/error.h).
std::string run(const MeasureError& command);

// `field` as one CSV field: as it is, or, if it holds a comma, a double
// quote or a line break, quoted with its quotes doubled (RFC 4180), so a
// path cannot split its row.
std::string csv_field(std::string_view field);

}  // namespace serenity::measure
