#include "headless/options.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace serenity::headless {

namespace {

constexpr std::string_view usage =
    "usage: serenity-headless --graph FILE [--scene FILE] --out DIRECTORY [--frames N] [--first I] "
    "[--step SECONDS | --time SECONDS] [--size WIDTHxHEIGHT] [--write all|last|doubling] [--samples N]";

// The whole of `text` as a T, or none: std::from_chars reads no locale and
// sets no errno (E.28), and takes no sign or space before the number.
template <typename T>
std::optional<T> parsed(std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

// Seconds from `text`: finite, and > 0 when `positive`, else >= 0.
double seconds(std::string_view option, std::string_view text, bool positive) {
    const std::optional<double> value = parsed<double>(text);
    if (!value || !std::isfinite(*value) || (positive ? *value <= 0.0 : *value < 0.0)) {
        throw OptionsError(std::string{option} +
                           (positive ? " needs a positive number of seconds, not '"
                                     : " needs a number of seconds, 0 or more, not '") +
                           std::string{text} + "'");
    }
    return *value;
}

std::uint64_t whole_number(std::string_view option, std::string_view text) {
    const std::optional<std::uint64_t> value = parsed<std::uint64_t>(text);
    if (!value) {
        throw OptionsError(std::string{option} + " needs a whole number, not '" + std::string{text} + "'");
    }
    return *value;
}

// One side of --size: a whole number from 1 to what a uint32 holds, or
// none.
std::optional<std::uint32_t> side(std::string_view text) {
    const std::optional<std::uint64_t> value = parsed<std::uint64_t>(text);
    if (!value || *value == 0 || *value > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(*value);
}

// The arguments in order, each value taken by the option before it: a
// cursor, so no loop index is stepped inside the loop's body (ES.86).
class Arguments {
public:
    explicit Arguments(std::span<const char* const> args) : args_(args) {}

    std::optional<std::string_view> next() {
        if (at_ == args_.size()) {
            return std::nullopt;
        }
        return args_[at_++];
    }

private:
    std::span<const char* const> args_;
    std::size_t at_ = 0;
};

}  // namespace

Options parse(std::span<const char* const> args) {
    Options options;
    bool stepped = false;
    Arguments arguments(args);
    while (const std::optional<std::string_view> next = arguments.next()) {
        const std::string_view option = *next;
        const auto value = [&] {
            const std::optional<std::string_view> given = arguments.next();
            if (!given) {
                throw OptionsError(std::string{option} + " needs a value");
            }
            return *given;
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
                throw OptionsError("--frames must be at least 1");
            }
        } else if (option == "--samples") {
            options.samples = whole_number(option, value());
            if (options.samples == 0 || options.samples > frame::max_accumulated_frames) {
                throw OptionsError("--samples must be from 1 to " + std::to_string(frame::max_accumulated_frames) +
                                   ", the most an image holds");
            }
        } else if (option == "--first") {
            options.first = whole_number(option, value());
        } else if (option == "--step") {
            options.step = frame::Seconds{seconds(option, value(), true)};
            stepped = true;
        } else if (option == "--time") {
            options.time = frame::Seconds{seconds(option, value(), false)};
        } else if (option == "--write") {
            const std::string_view which = value();
            if (which == "all") {
                options.write = Write::all;
            } else if (which == "last") {
                options.write = Write::last;
            } else if (which == "doubling") {
                options.write = Write::doubling;
            } else {
                throw OptionsError("--write needs all, last or doubling, not '" + std::string{which} + "'");
            }
        } else if (option == "--size") {
            const std::string_view text = value();
            const std::size_t x = text.find('x');
            const std::optional<std::uint32_t> width =
                x == std::string_view::npos ? std::nullopt : side(text.substr(0, x));
            const std::optional<std::uint32_t> height =
                x == std::string_view::npos ? std::nullopt : side(text.substr(x + 1));
            if (!width || !height) {
                throw OptionsError("--size needs WIDTHxHEIGHT, each a whole number at least 1, not '" +
                                   std::string{text} + "'");
            }
            options.size = frame::Extent{*width, *height};
        } else {
            throw OptionsError("unknown option '" + std::string{option} + "'; " + std::string{usage});
        }
    }
    if (options.graph.empty()) {
        throw OptionsError("missing --graph; " + std::string{usage});
    }
    if (options.out.empty()) {
        throw OptionsError("missing --out; " + std::string{usage});
    }
    if (stepped && options.time) {
        throw OptionsError("--time and --step are exclusive: --time freezes every frame at one time");
    }
    // The last frame is first + frames - 1, which must exist; first + frames
    // need not (frames is at least 1 here).
    if (options.first > std::numeric_limits<std::uint64_t>::max() - (options.frames - 1)) {
        throw OptionsError("--first plus --frames is past the last frame there can be");
    }
    // Its last sample, last x N + N - 1, must exist too.
    const std::uint64_t last = options.first + (options.frames - 1);
    const std::uint64_t n = options.samples;
    if (last > (std::numeric_limits<std::uint64_t>::max() - (n - 1)) / n) {
        throw OptionsError("--first plus --frames, at --samples per frame, is past the last sample there can be");
    }
    return options;
}

bool written(Write write, RunFrame frame) {
    const std::uint64_t n = frame.after_first;
    // No default: a way of writing without a rule fails to compile.
    switch (write) {
    case Write::all:
        return true;
    case Write::last:
        return n + 1 == frame.frames;
    case Write::doubling:
        // The 1st, 2nd, 4th ... frame, counting from 1: n + 1 a power of two.
        return n + 1 == frame.frames || ((n + 1) & n) == 0;
    }
    // Reached only by a value no enumerator names: not a reason to write.
    throw std::logic_error("written: a way of writing with no rule");
}

void prepare_output(const std::filesystem::path& out) {
    if (std::filesystem::create_directories(out)) {
        return;  // made just now, empty
    }
    if (!std::filesystem::is_directory(out)) {
        throw OptionsError("--out " + out.string() + " is not a directory");
    }
    if (!std::filesystem::is_empty(out)) {
        throw OptionsError("--out " + out.string() +
                           " already holds files, which would sit beside this run's frames as if they were its own; "
                           "empty it or name another");
    }
}

}  // namespace serenity::headless
