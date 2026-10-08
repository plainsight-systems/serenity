#include "app/options.h"

#include <string_view>

namespace serenity::app {

Options parse(std::span<const char* const> args) {
    Options options;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--graph") {
            if (i + 1 >= args.size()) {
                throw OptionsError("--graph needs a file");
            }
            options.graph = args[++i];
        } else {
            throw OptionsError("unknown option '" + std::string(arg) + "'; usage: serenity --graph FILE");
        }
    }
    if (options.graph.empty()) {
        throw OptionsError("missing --graph; usage: serenity --graph FILE");
    }
    return options;
}

}  // namespace serenity::app
