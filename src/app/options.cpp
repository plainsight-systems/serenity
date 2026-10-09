#include "app/options.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace serenity::app {

namespace {

constexpr const char* usage = "usage: serenity --graph FILE [--scene FILE] [--scale S]";

}  // namespace

Options parse(std::span<const char* const> args) {
    Options options;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--graph") {
            if (i + 1 >= args.size()) {
                throw OptionsError("--graph needs a file");
            }
            options.graph = args[++i];
        } else if (arg == "--scene") {
            if (i + 1 >= args.size()) {
                throw OptionsError("--scene needs a file");
            }
            options.scene = args[++i];
        } else if (arg == "--scale") {
            if (i + 1 >= args.size()) {
                throw OptionsError("--scale needs a number");
            }
            const char* text = args[++i];
            errno = 0;
            char* end = nullptr;
            const double scale = std::strtod(text, &end);
            if (errno != 0 || end == text || *end != '\0' || !std::isfinite(scale) || scale <= 0.0 || scale > 1.0) {
                throw OptionsError(std::string("--scale needs a number in (0, 1], not '") + text + "'");
            }
            options.scale = scale;
        } else {
            throw OptionsError("unknown option '" + std::string(arg) + "'; " + usage);
        }
    }
    if (options.graph.empty()) {
        throw OptionsError(std::string("missing --graph; ") + usage);
    }
    return options;
}

frame::Extent render_size(frame::Extent window, double scale) {
    const auto side = [scale](std::uint32_t pixels) {
        return static_cast<std::uint32_t>(std::max(1.0, std::round(pixels * scale)));
    };
    return frame::Extent{side(window.width), side(window.height)};
}

}  // namespace serenity::app
