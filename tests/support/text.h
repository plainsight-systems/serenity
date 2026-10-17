#pragma once

// What the tests share for text and what calls throw (ES.3, F.10): the
// message a call throws, or the thrown object itself; a text with one part
// replaced; and whether a text holds a part.

#include <concepts>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <doctest/doctest.h>

namespace serenity::tests {

// What `call` throws as an E, its what(); "" if it throws nothing. Anything
// else it throws passes on, and fails the test case.
template <typename E, std::invocable F>
std::string error_of(F&& call) {
    try {
        (void)std::forward<F>(call)();
    } catch (const E& error) {
        return error.what();
    }
    return "";
}

// What `call` throws as an E, a copy of it, for a test that reads the
// thrown object's own fields; none if it throws nothing. Anything else it
// throws passes on, and fails the test case, as error_of()'s does.
template <std::copy_constructible E, std::invocable F>
std::optional<E> thrown(F&& call) {
    try {
        (void)std::forward<F>(call)();
    } catch (const E& error) {
        return error;
    }
    return std::nullopt;
}

// Whether `text` holds `part`.
inline bool contains(std::string_view text, std::string_view part) {
    return text.find(part) != std::string_view::npos;
}

// `text` with its first `from` replaced by `to`. `from` must be in it: a
// replacement that silently did nothing would test the text unchanged.
inline std::string replaced(std::string text, std::string_view from, std::string_view to) {
    const std::size_t at = text.find(from);
    INFO("replacing '" << from << "'");
    REQUIRE(at != std::string::npos);
    return text.replace(at, from.size(), to);
}

}  // namespace serenity::tests
