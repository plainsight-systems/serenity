# Writes a .metallib's bytes as a C++ array (cmake/MetalLibrary.cmake).
#
#   cmake -DINPUT=<.metallib> -DOUTPUT=<header> -DNAME=<identifier> -P <this>
#
# The header holds serenity::metallib::<NAME>, the file's bytes as a span of
# std::byte over an inline constexpr array of them, sixteen to a line (the
# array of unsigned char a literal can spell). A missing or empty input is a build
# failure: an empty array would load as a library Metal rejects at run time,
# which is later and further from the cause.

if(NOT EXISTS "${INPUT}")
    message(FATAL_ERROR "embed_metallib: ${INPUT} does not exist")
endif()
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" digits)
if(digits EQUAL 0)
    message(FATAL_ERROR "embed_metallib: ${INPUT} is empty")
endif()
math(EXPR size "${digits} / 2")

string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){16})" "\\1\n    " bytes "${bytes}")

get_filename_component(source_name "${INPUT}" NAME)
file(WRITE "${OUTPUT}"
"// Generated from ${source_name} by cmake/embed_metallib_script.cmake. Do not edit.
#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace serenity::metallib {

namespace stored {

// ${size} bytes, as a std::array (SL.con.1) of what a literal can spell.
inline constexpr std::array<unsigned char, ${size}> ${NAME} = {
    ${bytes}
};

}  // namespace stored

// The same bytes as bytes (SL.str.5), as metal::Library takes them.
inline const std::span<const std::byte> ${NAME} = std::as_bytes(std::span(stored::${NAME}));

}  // namespace serenity::metallib
")
