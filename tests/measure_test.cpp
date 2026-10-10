// serenity-measure (measure/options.h): its command line, every mistake
// refused by name; and the program run as its user runs it, on PFMs this
// test writes: the reference's file and its printed floors, the error's
// CSV, and a run that fails printing no result, partial or whole.

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

TEST_CASE("measure: a reference written, and its floors printed, from the batches given") {
    const ScratchDirectory scratch("measure-reference");
    const std::vector<LinearImage> batches{image(1.0f), image(1.5f), image(0.75f)};
    serenity::measurement::ReferenceBuilder expected;
    std::vector<std::string> args{"reference", "--out", (scratch / "reference.pfm").string()};
    for (std::size_t b = 0; b < batches.size(); ++b) {
        const std::filesystem::path path = scratch / ("batch-" + std::to_string(b) + ".pfm");
        serenity::output::write_pfm(path, batches[b]);
        expected.add(batches[b]);
        args.push_back(path.string());
    }
    const Ran ran = measure(args, scratch);
    CHECK(ran.status == 0);
    CHECK(ran.err.empty());
    CHECK(ran.out == std::format("batches 3\nmse_floor {:.9g}\nrelative_mse_floor {:.9g}\n", expected.mse_floor(),
                                 expected.relative_mse_floor()));
    CHECK(serenity::output::read_pfm(scratch / "reference.pfm").rgb == expected.mean().rgb);

    // Asked again, its file there: refused, the file as it was.
    const std::string before = serenity::tests::read_bytes(scratch / "reference.pfm");
    check_failed(measure(args, scratch), "already exists");
    CHECK(serenity::tests::read_bytes(scratch / "reference.pfm") == before);
}

TEST_CASE("measure: the error's CSV, a row an image in the order given, a path with a comma quoted") {
    const ScratchDirectory scratch("measure-error");
    const LinearImage reference = image(1.0f);
    serenity::output::write_pfm(scratch / "reference.pfm", reference);
    const std::filesystem::path near = scratch / "near.pfm";
    const std::filesystem::path far = scratch / "far, \"quoted\".pfm";
    serenity::output::write_pfm(near, image(1.125f));
    serenity::output::write_pfm(far, image(2.0f));

    const Ran ran = measure({"error", "--reference", (scratch / "reference.pfm").string(), far.string(), near.string()},
                            scratch);
    CHECK(ran.status == 0);
    CHECK(ran.err.empty());
    const auto row = [&](const std::string& field, const LinearImage& judged) {
        const auto error = serenity::measurement::error_against({.image = judged, .reference = reference});
        return std::format("{},{:.9g},{:.9g}\n", field, error.mse, error.relative_mse);
    };
    // RFC 4180: the field quoted, each quote in it doubled.
    std::string far_field = "\"";
    for (const char c : far.string()) {
        far_field += c == '"' ? std::string{"\"\""} : std::string{c};
    }
    far_field += "\"";
    CHECK(contains(far_field, "far, \"\"quoted\"\".pfm\""));
    CHECK(ran.out == "image,mse,relative_mse\n" + row(far_field, image(2.0f)) + row(near.string(), image(1.125f)));
}

TEST_CASE("measure: a file that fails ends the run with nothing printed as a result") {
    const ScratchDirectory scratch("measure-failing");
    serenity::output::write_pfm(scratch / "reference.pfm", image(1.0f));
    serenity::output::write_pfm(scratch / "good.pfm", image(1.25f));
    serenity::tests::write_bytes(scratch / "bad.pfm", "Pf\n2 2\n-1.0\n");
    serenity::output::write_pfm(scratch / "small.pfm", LinearImage{Extent{1, 1}, {1.0f, 1.0f, 1.0f}});
    const std::string reference = (scratch / "reference.pfm").string();
    const std::string good = (scratch / "good.pfm").string();

    // The good image's row is never printed: the last file fails.
    check_failed(measure({"error", "--reference", reference, good, (scratch / "bad.pfm").string()}, scratch),
                 "bad.pfm");
    check_failed(measure({"error", "--reference", reference, good, (scratch / "none.pfm").string()}, scratch),
                 "none.pfm");
    check_failed(measure({"error", "--reference", reference, good, (scratch / "small.pfm").string()}, scratch),
                 "the image is 1 x 1, the reference 2 x 2");
    check_failed(measure({"error", "--reference", (scratch / "bad.pfm").string(), good}, scratch), "bad.pfm");

    // A batch that fails: no reference written, no floor printed.
    const std::filesystem::path out = scratch / "made.pfm";
    check_failed(measure({"reference", "--out", out.string(), good, (scratch / "bad.pfm").string()}, scratch),
                 "bad.pfm");
    check_failed(measure({"reference", "--out", out.string(), good, (scratch / "small.pfm").string()}, scratch),
                 "where the first batch is 2 x 2");
    CHECK_FALSE(std::filesystem::exists(out));
    // And a command line refused, before any file is read.
    check_failed(measure({"reference", "--out", out.string(), good}, scratch), "two batches or more");
}
