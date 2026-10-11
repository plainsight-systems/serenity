// serenity-measure: a reference made from the headless renderer's batches,
// and images judged against it, on the CPU (measure/options.h). Every file
// is read, and every number computed, before anything is printed as a
// result, and what is printed is printed once: a run that fails prints
// nothing but its message, on stderr, with status 1 (E.2).

#include <cstddef>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/contracts/linear_image.h"
#include "core/measurement/error.h"
#include "core/measurement/reference.h"
#include "core/output/pfm.h"
#include "measure/options.h"

namespace serenity::measure {

namespace {

// The arguments after the program's name; none if the system gave not even
// that (POSIX allows argc == 0).
std::span<const char* const> arguments(int argc, char** argv) {
    if (argc < 1) {
        return {};
    }
    return {argv + 1, static_cast<std::size_t>(argc - 1)};
}

// `field` as one CSV field: as it is, or, if it holds a comma, a double
// quote or a line break, quoted with its quotes doubled (RFC 4180), so a
// path cannot split its row (measure/options.h).
std::string csv_field(std::string_view field) {
    if (field.find_first_of(",\"\r\n") == std::string_view::npos) {
        return std::string{field};
    }
    std::string quoted = "\"";
    for (const char c : field) {
        quoted += c;
        if (c == '"') {
            quoted += '"';
        }
    }
    return quoted + "\"";
}

// What a command prints, made whole before any of it is printed.
std::string run(const MakeReference& command) {
    measurement::ReferenceBuilder builder;
    for (const std::filesystem::path& batch : command.batches) {
        // One batch held at a time: read, folded in, dropped (reference.h).
        builder.add(output::read_pfm(batch));
    }
    // The floors before the file, so nothing after the write can fail but
    // printing.
    const double mse_floor = builder.mse_floor();
    const double relative_mse_floor = builder.relative_mse_floor();
    // Created, never written over: the exclusive create is the check that
    // --out names nothing, so two runs racing to one file cannot both write
    // it (measure/options.h, core/output/pfm.h).
    output::write_pfm(command.out, builder.mean());
    return std::format("batches {}\nmse_floor {:.9g}\nrelative_mse_floor {:.9g}\n", builder.batches(), mse_floor,
                       relative_mse_floor);
}

std::string run(const MeasureError& command) {
    const contracts::LinearImage reference = output::read_pfm(command.reference);
    std::string table = "image,mse,relative_mse\n";
    for (const std::filesystem::path& image : command.images) {
        const measurement::ImageError error =
            measurement::error_against({.image = output::read_pfm(image), .reference = reference});
        table += std::format("{},{:.9g},{:.9g}\n", csv_field(image.string()), error.mse, error.relative_mse);
    }
    return table;
}

}  // namespace

}  // namespace serenity::measure

int main(int argc, char** argv) {
    namespace measure = serenity::measure;
    try {
        const measure::Command command = measure::parse(measure::arguments(argc, argv));
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
