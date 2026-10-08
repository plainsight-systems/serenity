#include "headless/options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace serenity::headless {

namespace {

constexpr const char* usage =
    "usage: serenity-headless --scene FILE --out DIRECTORY [--frames N] [--first I] [--step SECONDS] "
    "[--size WIDTHxHEIGHT]";

std::uint64_t whole_number(std::string_view option, const char* text) {
    // strtoull accepts a sign and leading space; neither is a frame count.
    if (text[0] < '0' || text[0] > '9') {
        throw Error(std::string(option) + " needs a whole number, not '" + text + "'");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0') {
        throw Error(std::string(option) + " needs a whole number, not '" + text + "'");
    }
    return value;
}

std::uint32_t side(std::string_view option, std::string_view text, std::string_view whole) {
    const std::string copy(text);
    const std::uint64_t value = whole_number(option, copy.c_str());
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
        throw Error(std::string(option) + " needs WIDTHxHEIGHT, each at least 1, not '" + std::string(whole) + "'");
    }
    return static_cast<std::uint32_t>(value);
}

}  // namespace

Options parse(std::span<const char* const> args) {
    Options options;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view option = args[i];
        const auto value = [&]() -> const char* {
            if (i + 1 >= args.size()) {
                throw Error(std::string(option) + " needs a value");
            }
            return args[++i];
        };

        if (option == "--scene") {
            options.scene = value();
        } else if (option == "--out") {
            options.out = value();
        } else if (option == "--frames") {
            options.frames = whole_number(option, value());
            if (options.frames == 0) {
                throw Error("--frames must be at least 1");
            }
        } else if (option == "--first") {
            options.first = whole_number(option, value());
        } else if (option == "--step") {
            const char* text = value();
            errno = 0;
            char* end = nullptr;
            const double seconds = std::strtod(text, &end);
            if (errno != 0 || end == text || *end != '\0' || !std::isfinite(seconds) || seconds <= 0.0) {
                throw Error(std::string("--step needs a positive number of seconds, not '") + text + "'");
            }
            options.step = frame::Seconds(seconds);
        } else if (option == "--size") {
            const std::string_view text = value();
            const std::size_t x = text.find('x');
            if (x == std::string_view::npos) {
                throw Error("--size needs WIDTHxHEIGHT, not '" + std::string(text) + "'");
            }
            options.size = frame::Extent{side(option, text.substr(0, x), text),
                                         side(option, text.substr(x + 1), text)};
        } else {
            throw Error("unknown option '" + std::string(option) + "'; " + usage);
        }
    }
    if (options.scene.empty()) {
        throw Error(std::string("missing --scene; ") + usage);
    }
    if (options.out.empty()) {
        throw Error(std::string("missing --out; ") + usage);
    }
    if (options.first > std::numeric_limits<std::uint64_t>::max() - options.frames) {
        throw Error("--first plus --frames is past the last frame there can be");
    }
    return options;
}

}  // namespace serenity::headless
