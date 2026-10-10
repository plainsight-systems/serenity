#include "measure/options.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace serenity::measure {

namespace {

constexpr std::string_view usage =
    "usage: serenity-measure reference --out FILE BATCH BATCH... | serenity-measure error --reference FILE IMAGE...; "
    "every file a PFM, and the reference's FILE new";

// The arguments in order, each value taken by the option before it: a
// cursor, so no loop index is stepped inside the loop's body (ES.86), as
// the headless renderer's (headless/options.cpp).
class Arguments {
public:
    explicit Arguments(std::span<const char* const> args) : args_(args) {}

    std::optional<std::string_view> next() {
        if (at_ == args_.size()) {
            return std::nullopt;
        }
        return args_[at_++];
    }

    // The value of `option`, the next argument; OptionsError if there is none.
    std::string_view value_of(std::string_view option) {
        const std::optional<std::string_view> given = next();
        if (!given) {
            throw OptionsError(std::string{option} + " needs a value");
        }
        return *given;
    }

private:
    std::span<const char* const> args_;
    std::size_t at_ = 0;
};

// Whether `argument` is an option, not a file: anything that starts "--".
bool is_option(std::string_view argument) {
    return argument.starts_with("--");
}

// A command's arguments: the one option it takes, `name`, exactly once, and
// its files in the order given. Any other option is refused by name.
struct Parsed {
    std::optional<std::filesystem::path> option;
    std::vector<std::filesystem::path> files;
};

Parsed parse_command(Arguments& arguments, std::string_view command, std::string_view name) {
    Parsed parsed;
    while (const std::optional<std::string_view> next = arguments.next()) {
        const std::string_view argument = *next;
        if (argument == name) {
            if (parsed.option) {
                throw OptionsError(std::string{name} + " is given twice; " + std::string{command} + " takes one");
            }
            parsed.option = std::filesystem::path{arguments.value_of(name)};
        } else if (is_option(argument)) {
            throw OptionsError("unknown option '" + std::string{argument} + "' for " + std::string{command} + "; " +
                               std::string{usage});
        } else {
            parsed.files.emplace_back(argument);
        }
    }
    if (!parsed.option) {
        throw OptionsError("missing " + std::string{name} + "; " + std::string{usage});
    }
    return parsed;
}

}  // namespace

Command parse(std::span<const char* const> args) {
    Arguments arguments(args);
    const std::optional<std::string_view> command = arguments.next();
    if (!command) {
        throw OptionsError("missing a command, reference or error; " + std::string{usage});
    }
    if (*command == "reference") {
        Parsed parsed = parse_command(arguments, *command, "--out");
        if (parsed.files.size() < 2) {
            throw OptionsError("reference needs two batches or more, so the reference has a floor; " +
                               std::to_string(parsed.files.size()) + " given");
        }
        return MakeReference{.out = std::move(*parsed.option), .batches = std::move(parsed.files)};
    }
    if (*command == "error") {
        Parsed parsed = parse_command(arguments, *command, "--reference");
        if (parsed.files.empty()) {
            throw OptionsError("error needs an image or more to judge; none given");
        }
        return MeasureError{.reference = std::move(*parsed.option), .images = std::move(parsed.files)};
    }
    throw OptionsError("unknown command '" + std::string{*command} + "'; " + std::string{usage});
}

}  // namespace serenity::measure
