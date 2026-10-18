// serenity-measure (measure/options.h, commands.h): its command line,
// every mistake refused by name; its commands on PFMs this test writes, the
// reference's file and its floors, the error's CSV, a failing file refusing
// the whole; and, once, the program run as its user runs it, an integration
// test: a result printed, and a failure's message with status 1 and nothing
// printed as a result.

#include <cstddef>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <string>
#include <variant>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/linear_image.h"
#include "core/measurement/error.h"
#include "core/measurement/reference.h"
#include "core/output/pfm.h"
#include "measure/commands.h"
#include "measure/options.h"
#include "programs.h"
#include "support/files.h"
#include "support/run_program.h"
#include "support/text.h"

using serenity::contracts::LinearImage;
using serenity::frame::Extent;
using serenity::tests::contains;
using serenity::tests::Ran;
using serenity::tests::ScratchDirectory;

namespace {

serenity::measure::Command parsed(std::initializer_list<const char*> args) {
    return serenity::measure::parse(std::vector<const char*>(args));
}

std::string refusal(std::initializer_list<const char*> args) {
    return serenity::tests::error_of<serenity::measure::OptionsError>([&] { return parsed(args); });
}

// A 2 x 2 image, value i being `base` + i / 4.
LinearImage image(float base) {
    LinearImage made{Extent{2, 2}, std::vector<float>(12)};
    for (std::size_t i = 0; i < made.rgb.size(); ++i) {
        made.rgb[i] = base + static_cast<float>(i) * 0.25f;
    }
    return made;
}

Ran measure(const std::vector<std::string>& args, const ScratchDirectory& scratch) {
    return serenity::tests::run_program(serenity::tests::measure_program, args, scratch.path());
}

// What a failed run must look like: status 1, nothing on stdout, its
// message on stderr naming the program and `part`.
void check_failed(const Ran& ran, const std::string& part) {
    CHECK(ran.status == 1);
    CHECK(ran.out.empty());
    CHECK(ran.err.starts_with("serenity-measure: "));
    CHECK(contains(ran.err, part));
}

}  // namespace

TEST_CASE("measure: each command and its arguments, in the order given") {
    const auto reference =
        std::get<serenity::measure::MakeReference>(parsed({"reference", "--out", "r.pfm", "a", "b", "c"}));
    CHECK(reference.out == "r.pfm");
    CHECK(reference.batches == std::vector<std::filesystem::path>{"a", "b", "c"});
    // The option anywhere among the files.
    const auto later = std::get<serenity::measure::MakeReference>(parsed({"reference", "b", "a", "--out", "r.pfm"}));
    CHECK(later.batches == std::vector<std::filesystem::path>{"b", "a"});

    const auto error = std::get<serenity::measure::MeasureError>(parsed({"error", "--reference", "r.pfm", "x"}));
    CHECK(error.reference == "r.pfm");
    CHECK(error.images == std::vector<std::filesystem::path>{"x"});
}

TEST_CASE("measure: every mistake is refused by name") {
    CHECK(contains(refusal({}), "missing a command"));
    CHECK(contains(refusal({"refrence", "--out", "r", "a", "b"}), "unknown command 'refrence'"));
    CHECK(contains(refusal({"reference", "a", "b"}), "missing --out"));
    CHECK(contains(refusal({"reference", "--out", "r", "--out", "s", "a", "b"}), "--out is given twice"));
    CHECK(contains(refusal({"reference", "a", "b", "--out"}), "--out needs a value"));
    CHECK(contains(refusal({"reference", "--out", "r"}), "two batches or more"));
    CHECK(contains(refusal({"reference", "--out", "r", "a"}), "1 given"));
    CHECK(contains(refusal({"reference", "--out", "r", "--reference", "x", "a", "b"}),
                   "unknown option '--reference' for reference"));
    CHECK(contains(refusal({"error", "x"}), "missing --reference"));
    CHECK(contains(refusal({"error", "--reference", "r", "--reference", "s", "x"}), "--reference is given twice"));
    CHECK(contains(refusal({"error", "x", "--reference"}), "--reference needs a value"));
    CHECK(contains(refusal({"error", "--reference", "r"}), "an image or more"));
    CHECK(contains(refusal({"error", "--reference", "r", "--out", "o", "x"}), "unknown option '--out' for error"));
}

TEST_CASE("measure: a reference written, and its floors returned, from the batches given; never over a file") {
    const ScratchDirectory scratch("measure-reference");
    const std::vector<LinearImage> batches{image(1.0f), image(1.5f), image(0.75f)};
    serenity::measurement::ReferenceBuilder expected;
    serenity::measure::MakeReference command{.out = scratch / "reference.pfm", .batches = {}};
    for (std::size_t b = 0; b < batches.size(); ++b) {
        const std::filesystem::path path = scratch / ("batch-" + std::to_string(b) + ".pfm");
        serenity::output::write_pfm(path, batches[b]);
        expected.add(batches[b]);
        command.batches.push_back(path);
    }
    CHECK(serenity::measure::run(command) == std::format("batches 3\nmse_floor {:.9g}\nrelative_mse_floor {:.9g}\n",
                                                         expected.mse_floor(), expected.relative_mse_floor()));
    CHECK(serenity::output::read_pfm(scratch / "reference.pfm").rgb == expected.mean().rgb);

    // Asked again, its file there: refused, the file as it was.
    const std::string before = serenity::tests::read_bytes(scratch / "reference.pfm");
    CHECK_THROWS_WITH(serenity::measure::run(command), doctest::Contains("something is already there"));
    CHECK(serenity::tests::read_bytes(scratch / "reference.pfm") == before);
}

TEST_CASE("measure: the error's CSV, a row an image in the order given, a path with a comma quoted") {
    using serenity::measure::csv_field;
    CHECK(csv_field("plain.pfm") == "plain.pfm");
    // RFC 4180: the field quoted, each quote in it doubled; a line break
    // quoted as a comma is.
    CHECK(csv_field("far, \"quoted\".pfm") == "\"far, \"\"quoted\"\".pfm\"");
    CHECK(csv_field("two\nlines") == "\"two\nlines\"");

    const ScratchDirectory scratch("measure-error");
    const LinearImage reference = image(1.0f);
    serenity::output::write_pfm(scratch / "reference.pfm", reference);
    const std::filesystem::path near = scratch / "near.pfm";
    const std::filesystem::path far = scratch / "far, \"quoted\".pfm";
    serenity::output::write_pfm(near, image(1.125f));
    serenity::output::write_pfm(far, image(2.0f));
    const auto row = [&](const std::filesystem::path& path, const LinearImage& judged) {
        const auto error = serenity::measurement::error_against({.image = judged, .reference = reference});
        return std::format("{},{:.9g},{:.9g}\n", csv_field(path.string()), error.mse, error.relative_mse);
    };
    CHECK(serenity::measure::run(serenity::measure::MeasureError{.reference = scratch / "reference.pfm",
                                                                 .images = {far, near}}) ==
          "image,mse,relative_mse\n" + row(far, image(2.0f)) + row(near, image(1.125f)));
}

TEST_CASE("measure: a file that fails refuses the whole command, a reference unwritten") {
    const ScratchDirectory scratch("measure-failing");
    serenity::output::write_pfm(scratch / "reference.pfm", image(1.0f));
    serenity::output::write_pfm(scratch / "good.pfm", image(1.25f));
    serenity::tests::write_bytes(scratch / "bad.pfm", "Pf\n2 2\n-1.0\n");
    serenity::output::write_pfm(scratch / "small.pfm", LinearImage{Extent{1, 1}, {1.0f, 1.0f, 1.0f}});
    const auto error = [&](const std::filesystem::path& reference, std::vector<std::filesystem::path> images) {
        return serenity::tests::error_of<std::exception>([&] {
            return serenity::measure::run(serenity::measure::MeasureError{.reference = reference, .images = images});
        });
    };
    const std::filesystem::path good = scratch / "good.pfm";
    CHECK(contains(error(scratch / "reference.pfm", {good, scratch / "bad.pfm"}), "bad.pfm"));
    CHECK(contains(error(scratch / "reference.pfm", {good, scratch / "none.pfm"}), "none.pfm"));
    CHECK(contains(error(scratch / "reference.pfm", {good, scratch / "small.pfm"}),
                   "the image is 1 x 1, the reference 2 x 2"));
    CHECK(contains(error(scratch / "bad.pfm", {good}), "bad.pfm"));

    // A batch that fails: no reference written.
    const std::filesystem::path out = scratch / "made.pfm";
    CHECK_THROWS_WITH(serenity::measure::run(serenity::measure::MakeReference{.out = out,
                                                                              .batches = {good, scratch / "bad.pfm"}}),
                      doctest::Contains("bad.pfm"));
    CHECK_THROWS_WITH(serenity::measure::run(
                          serenity::measure::MakeReference{.out = out, .batches = {good, scratch / "small.pfm"}}),
                      doctest::Contains("where the first batch is 2 x 2"));
    CHECK_FALSE(std::filesystem::exists(out));
}

TEST_CASE("measure: the program prints its command's result, or its failure's message alone, status 1") {
    const ScratchDirectory scratch("measure-program");
    serenity::output::write_pfm(scratch / "reference.pfm", image(1.0f));
    serenity::output::write_pfm(scratch / "good.pfm", image(1.25f));
    const std::string reference = (scratch / "reference.pfm").string();
    const std::string good = (scratch / "good.pfm").string();
    const Ran ran = measure({"error", "--reference", reference, good}, scratch);
    CHECK(ran.status == 0);
    CHECK(ran.err.empty());
    CHECK(ran.out == serenity::measure::run(serenity::measure::MeasureError{.reference = reference, .images = {good}}));
    // The last file fails: the good image's row is never printed.
    check_failed(measure({"error", "--reference", reference, good, (scratch / "none.pfm").string()}, scratch),
                 "none.pfm");
}
