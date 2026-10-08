#include "app/options.h"

#include <string_view>

namespace serenity::app {

Options parse(std::span<const char* const> args) {
    Options options;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--scene") {
            if (i + 1 >= args.size()) {
                throw OptionsError("--scene needs a file");
            }
            options.scene = args[++i];
        } else {
            throw OptionsError("unknown option '" + std::string(arg) + "'; usage: serenity --scene FILE");
        }
    }
    if (options.scene.empty()) {
        throw OptionsError("missing --scene; usage: serenity --scene FILE");
    }
    return options;
}

}  // namespace serenity::app
