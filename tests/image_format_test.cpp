// Output's list of kinds (core/output/image_format.h): each kind's name,
// extension and source; the names a diagnostic lists; and write_image()
// writing each kind by its own writer, refusing a kind given the other
// source before anything is written.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "core/contracts/linear_image.h"
#include "core/output/image_format.h"
#include "core/output/pfm.h"
#include "core/output/png.h"
#include "support/files.h"
#include "support/text.h"

using serenity::contracts::LinearImage;
using serenity::frame::Extent;
using serenity::output::DisplayedImage;
using serenity::output::ImageFormat;
using serenity::output::ImageSource;
using serenity::output::write_image;
using serenity::tests::contains;
using serenity::tests::read_bytes;
using serenity::tests::ScratchDirectory;

TEST_CASE("output: each kind by its name, its extension and its source") {
    using serenity::output::extension;
    using serenity::output::image_format_named;
    using serenity::output::source;
    CHECK(image_format_named("png") == ImageFormat::png);
    CHECK(image_format_named("pfm") == ImageFormat::pfm);
    for (const char* none : {"Png", "PFM", "exr", "", " pfm", "pfm "}) {
        INFO("'" << none << "'");
        CHECK_FALSE(image_format_named(none).has_value());
    }
    CHECK(extension(ImageFormat::png) == "png");
    CHECK(extension(ImageFormat::pfm) == "pfm");
    CHECK(source(ImageFormat::png) == ImageSource::displayed);
    CHECK(source(ImageFormat::pfm) == ImageSource::accumulated);
    CHECK(serenity::output::default_format == ImageFormat::png);
    // What a diagnostic lists: every name, in the list's order.
    CHECK(serenity::output::image_format_names() == "png or pfm");
    // A value no kind names is not reached here: the enumeration's two
    // kinds span one bit, so none of its values is unnamed, and casting one
    // in would be undefined behavior.
}

TEST_CASE("output: write_image writes each kind by its own writer") {
    const ScratchDirectory scratch("image-format-write");
    // A displayed frame as a PNG: the same bytes as png.h's writer's.
    const std::vector<std::uint8_t> rgba{10, 20, 30, 255, 40, 50, 60, 128};
    write_image(ImageFormat::png, scratch / "frame.png", DisplayedImage{Extent{2, 1}, rgba});
    serenity::output::write_png(scratch / "direct.png", Extent{2, 1}, rgba);
    CHECK(read_bytes(scratch / "frame.png") == read_bytes(scratch / "direct.png"));

    // An accumulated image as a PFM: the same bytes as pfm.h's writer's.
    const LinearImage image{Extent{2, 1}, {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f}};
    write_image(ImageFormat::pfm, scratch / "frame.pfm", image);
    serenity::output::write_pfm(scratch / "direct.pfm", image);
    CHECK(read_bytes(scratch / "frame.pfm") == read_bytes(scratch / "direct.pfm"));
}

TEST_CASE("output: a kind given the other source is refused, and nothing is written") {
    const ScratchDirectory scratch("image-format-source");
    const std::vector<std::uint8_t> rgba(2 * 1 * 4, 7);
    const LinearImage image{Extent{2, 1}, std::vector<float>(6, 1.0f)};
    CHECK(contains(serenity::tests::error_of<std::invalid_argument>([&] {
                       write_image(ImageFormat::pfm, scratch / "a.pfm", DisplayedImage{Extent{2, 1}, rgba});
                   }),
                   "a pfm is not written from a displayed frame"));
    CHECK(contains(serenity::tests::error_of<std::invalid_argument>(
                       [&] { write_image(ImageFormat::png, scratch / "a.png", image); }),
                   "a png is not written from an accumulated image"));
    CHECK(std::filesystem::is_empty(scratch.path()));
}
