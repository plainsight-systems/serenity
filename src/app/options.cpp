#include "app/options.h"

#include <string_view>

namespace serenity::app {

namespace {

constexpr const char* usage = "usage: serenity --graph FILE [--scene FILE]";

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
        } else {
            throw OptionsError("unknown option '" + std::string(arg) + "'; " + usage);
        }
    }
    if (options.graph.empty()) {
        throw OptionsError(std::string("missing --graph; ") + usage);
    }
    return options;
}

}  // namespace serenity::app
