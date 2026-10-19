#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace serenity::denoise {

// Axis: Pass (the denoiser's network, read from its export).
//
// JSON, read strictly, for the one file this program reads in it: the
// exported network's layout.json (training/export.py; network.h). RFC 8259's
// grammar, whole: objects, arrays, strings with their escapes, numbers,
// true, false and null; nothing after the value but whitespace. A number is
// kept as a double, which holds every integer the layout writes (offsets,
// counts and shapes, all far below 2^53) exactly. Written here, some 200
// lines, rather than a new dependency for one file read once at start-up.
//
// Anything else is a JsonError naming the byte offset: a file that is not
// JSON is refused, never half-read (SL.io.2). So is an object that names a
// member twice, which RFC 8259 leaves to the reader and at() would read by
// its first. \u escapes outside the ASCII range are refused too, as nothing
// in a layout writes them: a reader of them would be code no file exercises.
//
// Not performance-sensitive: one small file, once.

class JsonError : public std::runtime_error {
 public:
  explicit JsonError(const std::string& what) : std::runtime_error(what) {}
};

struct Json;
using JsonArray = std::vector<Json>;
using JsonObject =
    std::vector<std::pair<std::string, Json>>;  // in the file's order

struct Json {
  std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>
      value;

  // The member `key` of an object; throws JsonError, naming `key`, if this
  // is no object or has no such member.
  const Json& at(std::string_view key) const;
  // This as each kind; throws JsonError, naming `what`, if it is another.
  double number(std::string_view what) const;
  const std::string& string(std::string_view what) const;
  const JsonArray& array(std::string_view what) const;
};

// Reads `text` as one JSON value; throws JsonError if it is not exactly one.
Json parse_json(std::string_view text);

}  // namespace serenity::denoise
