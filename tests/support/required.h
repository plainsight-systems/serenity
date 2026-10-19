#pragma once

// The value an optional holds, for a test to read (ES.65, I.6): a test
// REQUIREs it first, which reports an empty one by its expression; reading
// through required() makes the read itself checked as well
// (bugprone-unchecked-optional-access), so an empty one, if a test reads
// it unREQUIREd, fails that test by a named exception rather than reading
// nothing.

#include <optional>
#include <stdexcept>

namespace serenity::tests {

template <typename T>
const T& required(const std::optional<T>& held) {
  if (!held.has_value()) {
    throw std::logic_error("a test read an empty std::optional");
  }
  return *held;
}

}  // namespace serenity::tests
