#include "headless/options.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace serenity::headless {

namespace {

constexpr const char* usage =
    "usage: serenity-headless --graph FILE [--scene FILE] --out DIRECTORY [--frames N] [--first I] "
    "[--step SECONDS | --time SECONDS] [--size WIDTHxHEIGHT] [--write all|last|doubling]";

// A number of seconds from `text`, finite, and at least `least` (exclusive
// when `positive`).
double seconds(std::string_view option, const char* text, bool positive) {
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    const bool in_range = positive ? value > 0.0 : value >= 0.0;
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value) || !in_range) {
        throw Error(std::string(option) + (positive ? " needs a positive number of seconds, not '"
                                                    : " needs a number of seconds, 0 or more, not '") +
                    text + "'");
    }
    return value;
}

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
    bool stepped = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view option = args[i];
        const auto value = [&]() -> const char* {
            if (i + 1 >= args.size()) {
                throw Error(std::string(option) + " needs a value");
            }
            return args[++i];
        };

        if (option == "--graph") {
            options.graph = value();
        } else if (option == "--scene") {
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
            options.step = frame::Seconds(seconds(option, value(), true));
            stepped = true;
        } else if (option == "--time") {
            options.time = frame::Seconds(seconds(option, value(), false));
        } else if (option == "--write") {
            const std::string_view which = value();
            if (which == "all") {
                options.write = Write::all;
            } else if (which == "last") {
                options.write = Write::last;
            } else if (which == "doubling") {
                options.write = Write::doubling;
            } else {
                throw Error("--write needs all, last or doubling, not '" + std::string(which) + "'");
            }
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
    if (options.graph.empty()) {
        throw Error(std::string("missing --graph; ") + usage);
    }
    if (options.out.empty()) {
        throw Error(std::string("missing --out; ") + usage);
    }
    if (stepped && options.time) {
        throw Error("--time and --step are exclusive: --time freezes every frame at one time");
    }
    // The last frame is first + frames - 1, which must exist; first + frames
    // need not (frames is at least 1 here).
    if (options.first > std::numeric_limits<std::uint64_t>::max() - (options.frames - 1)) {
        throw Error("--first plus --frames is past the last frame there can be");
    }
    return options;
}

bool written(Write write, std::uint64_t n, std::uint64_t frames) {
    // No default: a way of writing without a rule fails to compile.
    switch (write) {
    case Write::all:
        return true;
    case Write::last:
        return n + 1 == frames;
    case Write::doubling:
        // The 1st, 2nd, 4th ... frame, counting from 1: n + 1 a power of two.
        return n + 1 == frames || ((n + 1) & n) == 0;
    }
    return true;
}

}  // namespace serenity::headless
