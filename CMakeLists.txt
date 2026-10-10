cmake_minimum_required(VERSION 3.25)
# C and Objective-C as well as C++: SDL is built with them, so they are
# checked against the pin with the C++ compiler, below.
project(serenity LANGUAGES C CXX OBJC)

# ---------------------------------------------------------------------------
# One target for now: Apple silicon, through Metal. The Vulkan backend for the
# AMD machine comes later, behind the same core, as a backend of its own
# (docs/process/MEMORY.md).
# ---------------------------------------------------------------------------
if(NOT APPLE)
    message(FATAL_ERROR "serenity: only the Metal backend exists yet, which needs macOS.")
endif()
foreach(language CXX C OBJC)
    if(NOT CMAKE_${language}_COMPILER_ID STREQUAL "AppleClang")
        message(FATAL_ERROR
            "serenity: the ${language} compiler is ${CMAKE_${language}_COMPILER_ID} "
            "${CMAKE_${language}_COMPILER}, not Apple's.\n"
            "Configure through a preset (cmake --preset native-debug), which selects /usr/bin/clang and clang++.")
    endif()
endforeach()

# The toolchain must be the pinned one (cmake/toolchain.json). Checked here so
# that every way into the build is checked, not only the Makefile's: each
# compiler CMake resolved, the SDK it compiles against (the presets name it,
# CMAKE_OSX_SYSROOT; CMake 4 sets none by default, which leaves the SDK to
# the environment), and CMake itself. Python runs the check and builds
# nothing; it is found here, explicitly, rather than taken from PATH, and
# given a floor.
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_toolchain.py"
            --cxx "${CMAKE_CXX_COMPILER}"
            --cc "${CMAKE_C_COMPILER}"
            --objc "${CMAKE_OBJC_COMPILER}"
            --sdk "${CMAKE_OSX_SYSROOT}"
            --cmake "${CMAKE_COMMAND}"
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
# And in src/, beyond them: a conversion that can lose a value or change its
# sign must be written as a named cast (ES.46, ES.48, ES.49), not happen
# silently; a name may not hide another in an enclosing scope (ES.12); and a
# C-style cast, which can do any of static_cast, const_cast and
# reinterpret_cast without saying which, is refused (ES.49).
set(SERENITY_SOURCE_WARNINGS ${SERENITY_WARNINGS}
    -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast)

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
target_link_libraries(metal_cpp INTERFACE "-framework Metal" "-framework Foundation" "-framework QuartzCore")

# ---------------------------------------------------------------------------
# SDL3: the window (src/app/ only). Pinned by its release tarball's SHA-256:
# its git history is far larger than the release. Static, and only the parts
# the window uses: no OpenGL or OpenGL ES, which nothing here draws with.
# Vulkan stays, for the second backend's window.
# ---------------------------------------------------------------------------
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
foreach(subsystem AUDIO CAMERA HAPTIC JOYSTICK SENSOR HIDAPI GPU RENDER POWER DIALOG TRAY OPENGL OPENGLES)
    set(SDL_${subsystem} OFF CACHE BOOL "" FORCE)
endforeach()
FetchContent_Declare(
    SDL3
    URL      https://github.com/libsdl-org/SDL/releases/download/release-3.4.18/SDL3-3.4.18.tar.gz
    URL_HASH SHA256=9c75cf16330322c217dedd2e0609f1124f1b54b8633e763467b4684d0f4334a3
    SYSTEM  # its headers' warnings are its own, as the other dependencies' are
)
FetchContent_MakeAvailable(SDL3)

# toml++: reading frame graph and scene files (src/core/frame/graph_file.cpp and
# src/core/scene/scene.cpp only). v3.4.0.
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        30172438cee64926dc41fdd9c11fb3ba5b2ba9de  # v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

# stb_image_write: PNG output (src/core/output/png.cpp only). v1.16.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        2c980bb59875b0d32144a71867fbdebb2f77cd20
)
FetchContent_MakeAvailable(stb)
add_library(stb_image_write INTERFACE)
target_include_directories(stb_image_write SYSTEM INTERFACE "${stb_SOURCE_DIR}")

# ---------------------------------------------------------------------------
# Core: platform-neutral C++. Builds with no GPU API at all, so a second
# backend renders the same scene from the same code.
#
# Invariant: nothing in src/core/ uses a platform GPU or windowing API or
# includes from a backend, the app or the headless renderer.
# tools/check_boundaries.sh enforces it (make check).
# ---------------------------------------------------------------------------
add_library(serenity_core STATIC
    src/core/animation/animate.cpp
    src/core/animation/flight.cpp
    src/core/animation/glow.cpp
    src/core/animation/motion.cpp
    src/core/animation/wander.cpp
    src/core/camera/thin_lens.cpp
    src/core/materials/coated.cpp
    src/core/frame/graph_file.cpp
    src/core/frame/history.cpp
    src/core/frame/schedule.cpp
    src/core/measurement/frame_times.cpp
    src/core/output/png.cpp
    src/core/scene/scene.cpp
    src/core/scene/swarm.cpp
    src/core/shapes/shapes.cpp
)
target_include_directories(serenity_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(serenity_core PRIVATE tomlplusplus::tomlplusplus stb_image_write)
target_compile_options(serenity_core PRIVATE ${SERENITY_SOURCE_WARNINGS})

# ---------------------------------------------------------------------------
# Metal backend: the only code that uses Metal's host API. Its shaders are
# compiled into it (cmake/MetalLibrary.cmake), as serenity::metallib::shaders.
# ---------------------------------------------------------------------------
add_library(serenity_metal STATIC
    src/metal/device/metal_cpp.cpp
    src/metal/device/device.cpp
    src/metal/device/library.cpp
    src/metal/device/offscreen.cpp
    src/metal/device/presenter.cpp
    src/metal/device/frame_array.cpp
    src/metal/device/static_arrays.cpp
    src/metal/device/submission.cpp
    src/metal/device/support.cpp
    src/metal/acceleration/scene_acceleration.cpp
    src/metal/scene/light_glows.cpp
    src/metal/scene/scene_buffers.cpp
    src/metal/scene/shape_transforms.cpp
    src/metal/film/non_finite.cpp
    src/metal/frame/accumulation.cpp
    src/metal/frame/frame_images.cpp
    src/metal/frame/renderer.cpp
    src/metal/passes/display/display.cpp
    src/metal/passes/path/path.cpp
    src/metal/passes/preview/preview.cpp
    src/metal/passes/test_pattern/test_pattern.cpp
    src/metal/passes/tone_map/tone_map.cpp
)
target_link_libraries(serenity_metal PUBLIC serenity_core metal_cpp)
target_compile_options(serenity_metal PRIVATE ${SERENITY_SOURCE_WARNINGS})
serenity_add_metallib(
    TARGET serenity_metal NAME shaders
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/test_pattern/test_pattern.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/preview/preview.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/path/path.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/display/display.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/tone_map/tone_map.metal"
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# The two ways to run a frame. Each parses its command line in a library of
# its own, so the tests can reach it without a window or a GPU; the window's
# clock, which needs neither, is in its library too.
# ---------------------------------------------------------------------------
add_library(serenity_app_options STATIC src/app/options.cpp src/app/clock.cpp)
target_link_libraries(serenity_app_options PUBLIC serenity_core)
target_compile_options(serenity_app_options PRIVATE ${SERENITY_SOURCE_WARNINGS})

add_executable(serenity src/app/main.cpp src/app/window.cpp)
target_link_libraries(serenity PRIVATE serenity_app_options serenity_metal SDL3::SDL3-static)
target_compile_options(serenity PRIVATE ${SERENITY_SOURCE_WARNINGS})

add_library(serenity_headless_options STATIC src/headless/options.cpp)
target_link_libraries(serenity_headless_options PUBLIC serenity_core)
target_compile_options(serenity_headless_options PRIVATE ${SERENITY_SOURCE_WARNINGS})

add_executable(serenity-headless src/headless/main.cpp)
target_link_libraries(serenity-headless PRIVATE serenity_headless_options serenity_metal)
target_compile_options(serenity-headless PRIVATE ${SERENITY_SOURCE_WARNINGS})

enable_testing()
add_subdirectory(tests)
add_subdirectory(tests/gpu)
