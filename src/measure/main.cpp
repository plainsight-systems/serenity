// serenity-measure: a reference made from the headless renderer's batches,
// and images judged against it, on the CPU (measure/options.h). Every file
// is read, and every number computed, before anything is printed as a
// result (measure/commands.h), and what is printed is printed once: a run
// that fails prints nothing but its message, on stderr, with status 1 (E.2).

#include <cstddef>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>

#include "measure/commands.h"
#include "measure/options.h"

namespace {

// The arguments after the program's name; none if the system gave not even
// that (POSIX allows argc == 0).
std::span<const char* const> arguments(int argc, char** argv) {
    if (argc < 1) {
        return {};
    }
    return {argv + 1, static_cast<std::size_t>(argc - 1)};
}

}  // namespace

int main(int argc, char** argv) {
    namespace measure = serenity::measure;
    try {
        const measure::Command command = measure::parse(arguments(argc, argv));
        const std::string printed = std::visit([](const auto& which) { return measure::run(which); }, command);
        std::cout << printed << std::flush;
        if (!std::cout) {
            throw std::runtime_error("its result could not be written to stdout");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "serenity-measure: " << error.what() << '\n';
        return 1;
    }
}
