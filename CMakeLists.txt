cmake_minimum_required(VERSION 3.25)
project(serenity LANGUAGES CXX)

# ---------------------------------------------------------------------------
# One target for now: Apple silicon, through Metal. The Vulkan backend for the
# AMD machine comes later, behind the same core, as a backend of its own
# (docs/process/MEMORY.md).
# ---------------------------------------------------------------------------
if(NOT APPLE)
    message(FATAL_ERROR "serenity: only the Metal backend exists yet, which needs macOS.")
endif()
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(FATAL_ERROR
        "serenity: the compiler is ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}, not Apple's.\n"
        "Configure through a preset (cmake --preset native-debug), which selects /usr/bin/clang++.")
endif()

# The toolchain must be the pinned one (cmake/toolchain.json). Checked here so
# that every way into the build is checked, not only the Makefile's.
execute_process(
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_toolchain.py"
            --cxx "${CMAKE_CXX_COMPILER}"
    RESULT_VARIABLE toolchain_status)
if(NOT toolchain_status EQUAL 0)
    message(FATAL_ERROR "serenity: toolchain does not match cmake/toolchain.json (above).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set(SERENITY_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${SERENITY_CMAKE_DIR}")
include(MetalLibrary)
include(FetchContent)

# Warnings fail the build in this project's code. Third-party headers are
# included as SYSTEM, so their warnings are theirs.
set(SERENITY_WARNINGS -Wall -Wextra -Wpedantic -Werror)

# ---------------------------------------------------------------------------
# metal-cpp: Apple's C++ interface to Metal, header-only. Pinned by commit to
# the release for the pinned SDK (macOS 26, cmake/toolchain.json); a floating
# ref would make the build a moving target.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    metal_cpp
    GIT_REPOSITORY https://github.com/apple/metal-cpp.git
    GIT_TAG        f567ed836e4cbb85788c42115a2682bbe68097ee  # release/metal-cpp_macOS26_iOS26
)
FetchContent_MakeAvailable(metal_cpp)
add_library(metal_cpp INTERFACE)
target_include_directories(metal_cpp SYSTEM INTERFACE "${metal_cpp_SOURCE_DIR}")
target_link_libraries(metal_cpp INTERFACE "-framework Metal" "-framework Foundation")

# ---------------------------------------------------------------------------
# Core: platform-neutral C++. Builds with no GPU API at all, so a second
# backend renders the same scene from the same code.
#
# Invariant: nothing in src/core/ uses a platform GPU or windowing API or
# includes from a backend. tools/check_boundaries.sh enforces it (make check).
# ---------------------------------------------------------------------------
add_library(serenity_core INTERFACE)
target_include_directories(serenity_core INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# Metal backend: the only code that uses Metal's host API.
# ---------------------------------------------------------------------------
add_library(serenity_metal STATIC
    src/metal/metal_cpp.cpp
    src/metal/device.cpp
    src/metal/library.cpp
)
target_link_libraries(serenity_metal PUBLIC serenity_core metal_cpp)
target_compile_options(serenity_metal PRIVATE ${SERENITY_WARNINGS})

enable_testing()
add_subdirectory(tests)
add_subdirectory(tests/gpu)
