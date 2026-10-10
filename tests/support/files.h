#pragma once

// What the tests that write files share (ES.3, F.10): a directory of a
// test's own, and a file's bytes written and read whole.

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>

#include <unistd.h>

#include <doctest/doctest.h>

namespace serenity::tests {

// Under the system's temporary directory, named by the test and the process
// so two runs at once (two presets) never share one, empty when made and
// removed however the test ends (R.1).
class ScratchDirectory {
public:
    explicit ScratchDirectory(std::string_view name)
        : path_(std::filesystem::temp_directory_path() /
                ("serenity-" + std::string{name} + "-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~ScratchDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    ScratchDirectory(const ScratchDirectory&) = delete;
    ScratchDirectory& operator=(const ScratchDirectory&) = delete;
    ScratchDirectory(ScratchDirectory&&) = delete;
    ScratchDirectory& operator=(ScratchDirectory&&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }

    // `name` within it.
    std::filesystem::path operator/(std::string_view name) const { return path_ / name; }

private:
    std::filesystem::path path_;
};

// `bytes` as the whole of the file at `path`.
inline void write_bytes(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    REQUIRE(out.good());
}

// The whole of the file at `path`.
inline std::string read_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    return {std::istreambuf_iterator<char>(in), {}};
}

}  // namespace serenity::tests
