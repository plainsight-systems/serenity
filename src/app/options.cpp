#include "app/options.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace serenity::app {

namespace {

constexpr std::string_view usage = "usage: serenity --graph FILE [--scene FILE] [--scale S]";

// The whole of `text` as a number, or none: std::from_chars reads no locale
// and sets no errno (E.28), and takes no sign or space before the number.
std::optional<double> number(std::string_view text) {
    double value = 0.0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

bool is_scale(double scale) {
    return std::isfinite(scale) && scale > 0.0 && scale <= 1.0;
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
    Arguments arguments(args);
    while (const std::optional<std::string_view> arg = arguments.next()) {
        const auto value = [&](std::string_view what) {
            const std::optional<std::string_view> given = arguments.next();
            if (!given) {
                throw OptionsError(std::string{*arg} + " needs " + std::string{what});
            }
            return *given;
        };
        if (*arg == "--graph") {
            options.graph = value("a file");
        } else if (*arg == "--scene") {
            options.scene = value("a file");
        } else if (*arg == "--scale") {
            const std::string_view text = value("a number");
            const std::optional<double> scale = number(text);
            if (!scale || !is_scale(*scale)) {
                throw OptionsError("--scale needs a number in (0, 1], not '" + std::string{text} + "'");
            }
            options.scale = *scale;
        } else {
            throw OptionsError("unknown option '" + std::string{*arg} + "'; " + std::string{usage});
        }
    }
    if (options.graph.empty()) {
        throw OptionsError("missing --graph; " + std::string{usage});
    }
    return options;
}

frame::Extent render_size(frame::Extent window, double scale) {
    if (!is_scale(scale)) {
        throw OptionsError("render_size: the scale " + std::to_string(scale) + " is not in (0, 1]");
    }
    // In (0, 1], a scaled side is no longer than the window's: it fits.
    const auto side = [scale](std::uint32_t pixels) {
        return static_cast<std::uint32_t>(std::max(1.0, std::round(pixels * scale)));
    };
    return frame::Extent{side(window.width), side(window.height)};
}

}  // namespace serenity::app
