#pragma once

// A program run as its user runs it, from a test (ES.3, F.10): its
// arguments, each quoted for the shell, its stdout and stderr captured to
// files, and its exit status.

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <sys/wait.h>

#include <doctest/doctest.h>

#include "support/files.h"

namespace serenity::tests {

struct Ran {
    int status = -1;  // the exit status; -1 if it did not exit
    std::string out;  // what it printed on stdout
    std::string err;  // and on stderr
};

// `text` as one word for the shell: single-quoted, so nothing in it is
// expanded. The tests' paths and arguments hold no single quote, which is
// checked rather than escaped.
inline std::string shell_word(const std::string& text) {
    REQUIRE(text.find('\'') == std::string::npos);
    return "'" + text + "'";
}

// `program` run with `args`, its output captured in `scratch`.
inline Ran run_program(const std::filesystem::path& program, const std::vector<std::string>& args,
                       const std::filesystem::path& scratch) {
    const std::filesystem::path out = scratch / "program-stdout";
    const std::filesystem::path err = scratch / "program-stderr";
    std::string command = shell_word(program.string());
    for (const std::string& arg : args) {
        command += " " + shell_word(arg);
    }
    command += " >" + shell_word(out.string()) + " 2>" + shell_word(err.string());
    const int raw = std::system(command.c_str());
    Ran ran;
    ran.status = (raw != -1 && WIFEXITED(raw)) ? WEXITSTATUS(raw) : -1;
    ran.out = read_bytes(out);
    ran.err = read_bytes(err);
    return ran;
}

}  // namespace serenity::tests
