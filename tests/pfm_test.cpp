// Portable Float Maps (core/output/pfm.h): the exact bytes written, rows
// from the bottom; a file built here, by a writer of the test's own, read
// to the image it holds, top first; a round trip bit for bit, non-finite
// values included; and every file the reader refuses, each by name.

#include <array>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/resource.h>

#include <doctest/doctest.h>

#include "core/contracts/linear_image.h"
#include "core/output/pfm.h"
#include "support/files.h"
#include "support/text.h"

using serenity::contracts::LinearImage;
using serenity::frame::Extent;
using serenity::output::PfmError;
using serenity::output::read_pfm;
using serenity::output::write_pfm;
using serenity::tests::contains;
using serenity::tests::read_bytes;
using serenity::tests::ScratchDirectory;
using serenity::tests::write_bytes;

namespace {

// A 3 x 2 image whose every value differs: value i is 0.25 + i, so the
// top row holds 0.25 .. 8.25 and the bottom 9.25 .. 17.25. Not square, and
// not symmetric top to bottom, so a row unturned or width and height
// swapped shows.
LinearImage asymmetric() {
    LinearImage image{Extent{3, 2}, std::vector<float>(3 * 2 * 3)};
    for (std::size_t i = 0; i < image.rgb.size(); ++i) {
        image.rgb[i] = 0.25f + static_cast<float>(i);
    }
    return image;
}

// The test's own writer, independent of write_pfm: `header`, then each of
// `rows_bottom_first`'s floats as its 4 bytes on this (little-endian)
// machine, appended in the order given.
std::string pfm_bytes(std::string_view header, std::span<const float> rows_bottom_first) {
    std::string bytes{header};
    for (const float value : rows_bottom_first) {
        std::array<char, sizeof(float)> four{};
        std::memcpy(four.data(), &value, sizeof(float));
        bytes.append(four.data(), four.size());
    }
    return bytes;
}

// asymmetric()'s floats as the format stores them: the bottom row first.
std::vector<float> asymmetric_bottom_first() {
    const LinearImage image = asymmetric();
    std::vector<float> floats(image.rgb.begin() + 9, image.rgb.end());  // bottom row
    floats.insert(floats.end(), image.rgb.begin(), image.rgb.begin() + 9);  // top row
    return floats;
}

// Whether two images hold the same extent and the same bits, NaNs included.
bool same_bits(const LinearImage& a, const LinearImage& b) {
    return a.extent == b.extent && a.rgb.size() == b.rgb.size() &&
           std::memcmp(a.rgb.data(), b.rgb.data(), a.rgb.size() * sizeof(float)) == 0;
}

// What read_pfm() refuses the file of `bytes` with, or "".
std::string refusal_of(const ScratchDirectory& scratch, std::string_view bytes) {
    const std::filesystem::path path = scratch / "refused.pfm";
    write_bytes(path, bytes);
    return serenity::tests::error_of<PfmError>([&] { return read_pfm(path); });
}

}  // namespace

TEST_CASE("pfm: the bytes written are the header, then the rows from the bottom") {
    const ScratchDirectory scratch("pfm-written");
    write_pfm(scratch / "a.pfm", asymmetric());
    const std::string bytes = read_bytes(scratch / "a.pfm");
    const std::string header = "PF\n3 2\n-1.0\n";
    CHECK(bytes.substr(0, header.size()) == header);
    CHECK(bytes == pfm_bytes(header, asymmetric_bottom_first()));
}

TEST_CASE("pfm: a file from another writer reads to its image, rows from the top") {
    const ScratchDirectory scratch("pfm-foreign");
    write_bytes(scratch / "a.pfm", pfm_bytes("PF\n3 2\n-1.0\n", asymmetric_bottom_first()));
    CHECK(same_bits(read_pfm(scratch / "a.pfm"), asymmetric()));
    // pbrt-v4 writes its scale "%f": -1.000000, which is -1 exactly.
    write_bytes(scratch / "pbrt.pfm", pfm_bytes("PF\n3 2\n-1.000000\n", asymmetric_bottom_first()));
    CHECK(same_bits(read_pfm(scratch / "pbrt.pfm"), asymmetric()));
}

TEST_CASE("pfm: a round trip keeps every bit, non-finite values and an image one row high included") {
    const ScratchDirectory scratch("pfm-round-trip");
    LinearImage image = asymmetric();
    image.rgb[0] = std::numeric_limits<float>::quiet_NaN();
    image.rgb[4] = std::numeric_limits<float>::infinity();
    image.rgb[8] = -std::numeric_limits<float>::infinity();
    image.rgb[10] = -0.0f;
    image.rgb[13] = std::numeric_limits<float>::denorm_min();
    image.rgb[17] = -3.5f;
    write_pfm(scratch / "a.pfm", image);
    CHECK(same_bits(read_pfm(scratch / "a.pfm"), image));

    LinearImage row{Extent{4, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    write_pfm(scratch / "row.pfm", row);
    CHECK(same_bits(read_pfm(scratch / "row.pfm"), row));
}

TEST_CASE("pfm: an image the contract does not hold is refused before anything is written") {
    const ScratchDirectory scratch("pfm-write-refused");
    const std::filesystem::path path = scratch / "a.pfm";
    LinearImage short_of_one = asymmetric();
    short_of_one.rgb.pop_back();
    CHECK_THROWS_AS(write_pfm(path, short_of_one), std::invalid_argument);
    CHECK_THROWS_AS(write_pfm(path, LinearImage{Extent{0, 2}, {}}), std::invalid_argument);
    const auto past = static_cast<std::uint32_t>(serenity::contracts::max_image_side + 1);
    CHECK_THROWS_AS(write_pfm(path, LinearImage{Extent{past, 1}, std::vector<float>(std::size_t{past} * 3)}),
                    std::invalid_argument);
    CHECK_FALSE(std::filesystem::exists(path));
    // A path that cannot be created is PfmError, naming it.
    CHECK(contains(serenity::tests::error_of<PfmError>([&] { write_pfm(scratch / "no" / "a.pfm", asymmetric()); }),
                   "no/a.pfm: it cannot be created"));
}

TEST_CASE("pfm: a file is created, never written over: anything at the path is refused and left as it was") {
    const ScratchDirectory scratch("pfm-exclusive");
    const auto refusal = [&](const std::filesystem::path& path) {
        return serenity::tests::error_of<PfmError>([&] { write_pfm(path, asymmetric()); });
    };
    // Another writer's file, there before this write's open: the exclusive
    // create is the check, so a file that appears between any earlier
    // check and the write is refused just the same, and kept whole.
    write_bytes(scratch / "theirs.pfm", "another writer's file");
    CHECK(contains(refusal(scratch / "theirs.pfm"), "something is already there"));
    CHECK(read_bytes(scratch / "theirs.pfm") == "another writer's file");
    // An earlier PFM: refused too, not replaced by an image of the same size.
    write_pfm(scratch / "first.pfm", asymmetric());
    const std::string first = read_bytes(scratch / "first.pfm");
    CHECK(contains(refusal(scratch / "first.pfm"), "something is already there"));
    CHECK(read_bytes(scratch / "first.pfm") == first);
    // A dangling link: refused, and nothing made where it points.
    std::filesystem::create_symlink(scratch / "nowhere.pfm", scratch / "link.pfm");
    CHECK(contains(refusal(scratch / "link.pfm"), "something is already there"));
    CHECK_FALSE(std::filesystem::exists(scratch / "nowhere.pfm"));
    // A directory.
    std::filesystem::create_directory(scratch / "dir.pfm");
    CHECK(contains(refusal(scratch / "dir.pfm"), "something is already there"));
}

namespace {

// Files this process writes held to `bytes` for as long as it lives, and the
// signal a write past it raises ignored, so the write fails rather than
// ending the process; both restored however the test ends (R.1). Test-only
// process state: no other test runs beside it (doctest runs one at a time).
class FileSizeLimit {
public:
    explicit FileSizeLimit(rlim_t bytes) : handler_(std::signal(SIGXFSZ, SIG_IGN)) {
        REQUIRE(::getrlimit(RLIMIT_FSIZE, &saved_) == 0);
        rlimit limited = saved_;
        limited.rlim_cur = bytes;
        REQUIRE(::setrlimit(RLIMIT_FSIZE, &limited) == 0);
    }
    ~FileSizeLimit() {
        (void)::setrlimit(RLIMIT_FSIZE, &saved_);
        (void)std::signal(SIGXFSZ, handler_);
    }
    FileSizeLimit(const FileSizeLimit&) = delete;
    FileSizeLimit& operator=(const FileSizeLimit&) = delete;
    FileSizeLimit(FileSizeLimit&&) = delete;
    FileSizeLimit& operator=(FileSizeLimit&&) = delete;

private:
    rlimit saved_{};
    void (*handler_)(int);
};

}  // namespace

TEST_CASE("pfm: a write that fails once the file is made throws, and removes its own partial file") {
    const ScratchDirectory scratch("pfm-write-fails");
    const std::filesystem::path path = scratch / "big.pfm";
    // 64 x 64 x 3 floats, 48 KiB, past a limit of 1 KiB.
    const LinearImage big{Extent{64, 64}, std::vector<float>(std::size_t{64} * 64 * 3, 1.0f)};
    std::string why;
    {
        const FileSizeLimit limit(1024);
        why = serenity::tests::error_of<PfmError>([&] { write_pfm(path, big); });
    }
    CHECK(contains(why, "the write failed, and the partial file is removed"));
    CHECK_FALSE(std::filesystem::exists(path));
    // And with the limit gone, the same write succeeds.
    write_pfm(path, big);
    CHECK(read_pfm(path).rgb == big.rgb);
}

TEST_CASE("pfm: every malformed header is refused by name, before anything is allocated for it") {
    const ScratchDirectory scratch("pfm-read-refused");
    const std::vector<float> floats = asymmetric_bottom_first();
    const auto refused = [&](std::string_view header) { return refusal_of(scratch, pfm_bytes(header, floats)); };

    CHECK(contains(refused("Pf\n3 2\n-1.0\n"), "magic is 'Pf'"));  // greyscale
    CHECK(contains(refused("P6\n3 2\n-1.0\n"), "magic is 'P6'"));
    for (const std::string_view sides : {"-3 2", "+3 2", " 3 2", "3x 2", "0 2", "16385 2", "3 -2", "3 +2", "3  2",
                                         "3 2 ", "3 2x", "3 0", "3 16385", "3", "3,2", ""}) {
        INFO("size '" << sides << "'");
        CHECK(contains(refused("PF\n" + std::string{sides} + "\n-1.0\n"), "is not WIDTH HEIGHT"));
    }
    for (const std::string_view scale : {"1.0", "-2.0", "-0.5", "1", "0"}) {
        INFO("scale '" << scale << "'");
        CHECK(contains(refused("PF\n3 2\n" + std::string{scale} + "\n"), "not -1"));
    }
    for (const std::string_view scale : {"nan", "-nan", "x", "-1.0 ", " -1.0", ""}) {
        INFO("scale '" << scale << "'");
        const std::string why = refused("PF\n3 2\n" + std::string{scale} + "\n");
        CHECK((contains(why, "not a number") || contains(why, "not -1")));
    }
    // A line run past any header's length, and a header the file ends in.
    CHECK(contains(refusal_of(scratch, std::string(4096, 'P')), "longer than 32 bytes"));
    CHECK(contains(refusal_of(scratch, "PF\n3 2\n-1.0"), "ends in its header"));
    CHECK(contains(refusal_of(scratch, ""), "ends in its header"));
    // The largest image a header may claim, over a file of 3 floats: refused
    // by its size, not by a 3 GiB allocation.
    CHECK(contains(refused("PF\n16384 16384\n-1.0\n"), "truncated"));
}

TEST_CASE("pfm: data short of the header's or with anything after it is refused") {
    const ScratchDirectory scratch("pfm-read-length");
    const std::string whole = pfm_bytes("PF\n3 2\n-1.0\n", asymmetric_bottom_first());
    CHECK(contains(refusal_of(scratch, whole.substr(0, whole.size() - 1)), "truncated"));
    CHECK(contains(refusal_of(scratch, whole.substr(0, 12)), "truncated"));
    CHECK(contains(refusal_of(scratch, whole + "\n"), "1 bytes follow"));
    CHECK(contains(refusal_of(scratch, whole + whole), "follow"));
    CHECK(refusal_of(scratch, whole).empty());  // and the whole file is read
}

TEST_CASE("pfm: a file that is not there is PfmError naming it") {
    const ScratchDirectory scratch("pfm-missing");
    CHECK(contains(serenity::tests::error_of<PfmError>([&] { return read_pfm(scratch / "none.pfm"); }), "none.pfm"));
}
