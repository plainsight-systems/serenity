# Writes a .metallib's bytes as a C++ array (cmake/MetalLibrary.cmake).
#
#   cmake -DINPUT=<.metallib> -DOUTPUT=<header> -DNAME=<identifier> -P <this>
#
# The header holds serenity::metallib::<NAME>, an inline constexpr array of
# the file's bytes, sixteen to a line. A missing or empty input is a build
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

namespace serenity::metallib {

// ${size} bytes.
inline constexpr unsigned char ${NAME}[] = {
    ${bytes}
};

}  // namespace serenity::metallib
")
