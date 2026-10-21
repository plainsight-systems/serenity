cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# Two platforms, each with its backend (docs/research/2026-10-19-cuda-backend.md):
# macOS, Apple silicon, through Metal; Linux, the RTX 3090 Ti, through CUDA
# and OptiX, built a phase at a time. The
# presets are the same on both (CMakePresets.json), so `cmake --preset
# native-release` is one command everywhere; what only the Mac needs is
# set here, before the languages are enabled.
#
# On the Mac, C and Objective-C as well as C++: SDL is built with them, so
# they are checked against the pin with the C++ compiler, below. And
# Objective-C++, for the one file that names Core Animation's Objective-C
# types, the display link (src/metal/device/display_link.mm), which
# metal-cpp does not bind. Its SDK and deployment target, and the
# Objective-C compilers, Apple's as the presets' C and C++ are.
# ---------------------------------------------------------------------------
if(CMAKE_HOST_APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "")
    set(CMAKE_OSX_SYSROOT "macosx" CACHE STRING "")
    set(CMAKE_OBJC_COMPILER "/usr/bin/clang" CACHE FILEPATH "")
    set(CMAKE_OBJCXX_COMPILER "/usr/bin/clang++" CACHE FILEPATH "")
endif()
project(serenity LANGUAGES C CXX)
if(APPLE)
    # The Objective-C sources take the C and C++ flags the preset gives
    # (the sanitizer's, in native-sanitize).
    set(CMAKE_OBJC_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "")
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "")
    enable_language(OBJC OBJCXX)
    set(serenity_languages CXX C OBJC OBJCXX)
    set(serenity_compiler_id "AppleClang")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(serenity_languages CXX C)
    set(serenity_compiler_id "Clang")
else()
    message(FATAL_ERROR "serenity: builds on macOS (Metal) and Linux (CUDA), not ${CMAKE_SYSTEM_NAME}.")
endif()
foreach(language ${serenity_languages})
    if(NOT CMAKE_${language}_COMPILER_ID STREQUAL serenity_compiler_id)
        message(FATAL_ERROR
            "serenity: the ${language} compiler is ${CMAKE_${language}_COMPILER_ID} "
            "${CMAKE_${language}_COMPILER}, not ${serenity_compiler_id}.\n"
            "Configure through a preset (cmake --preset native-debug), which selects /usr/bin/clang and clang++.")
    endif()
endforeach()

# The toolchain must be the pinned one (cmake/toolchain.json). Checked here so
# that every way into the build is checked, not only the Makefile's: each
# compiler CMake resolved, the SDK it compiles against (the presets name it,
# CMAKE_OSX_SYSROOT; CMake 4 sets none by default, which leaves the SDK to
# the environment), and CMake itself. Python runs the check and builds
# nothing; it is found here, explicitly, rather than taken from PATH, and
# given a floor: 3.11, for tomllib, with which the data set's script reads
# its input graph and the demo's to hold them alike
# (scripts/make_denoiser_dataset.py, input_graph_mismatch).
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
set(toolchain_tools --cxx "${CMAKE_CXX_COMPILER}" --cc "${CMAKE_C_COMPILER}" --cmake "${CMAKE_COMMAND}")
if(APPLE)
    list(APPEND toolchain_tools --objc "${CMAKE_OBJC_COMPILER}" --objcxx "${CMAKE_OBJCXX_COMPILER}"
         --sdk "${CMAKE_OSX_SYSROOT}")
else()
    # The CUDA toolkit: its driver API's header and library, and nvcc, which
    # compiles the kernels (cmake/CudaModules.cmake). Not enable_language(CUDA),
    # which would hand host code to nvcc: the host code is the pinned clang's.
    find_package(CUDAToolkit 13.2 REQUIRED)
    list(APPEND toolchain_tools --nvcc "${CUDAToolkit_NVCC_EXECUTABLE}")
endif()
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_toolchain.py" ${toolchain_tools}
    RESULT_VARIABLE toolchain_status)
if(NOT toolchain_status EQUAL 0)
    message(FATAL_ERROR "serenity: toolchain does not match cmake/toolchain.json (above).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(APPLE)
    set(CMAKE_OBJCXX_STANDARD 20)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set(SERENITY_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${SERENITY_CMAKE_DIR}")
if(APPLE)
    include(MetalLibrary)
endif()
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

# The Mac's: metal-cpp, for the Metal backend, and SDL, for its window.
if(APPLE)
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
endif()

# ---------------------------------------------------------------------------
# SDL3: the window (src/app/ only), on both platforms. Pinned by its release
# tarball's SHA-256: its git history is far larger than the release. Static,
# and only the parts the window uses: no OpenGL or OpenGL ES, which nothing
# here draws with. On Linux its X11 driver alone, no Wayland or KMS: the
# RTX 3090 Ti's monitor runs an X server for the window's run and nothing
# else (docs/research/2026-10-20-cuda-window.md); Vulkan, which the CUDA
# backend presents through, on both.
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(driver WAYLAND KMSDRM)
        set(SDL_${driver} OFF CACHE BOOL "" FORCE)
    endforeach()
    set(SDL_X11 ON CACHE BOOL "" FORCE)
    # X11's extensions the window does not use (cursors, input devices,
    # display modes, shapes, screensavers, sync), whose headers the machine
    # lacks: SDL asks that each be named off.
    foreach(extension XCURSOR XDBE XINPUT XFIXES XRANDR XSCRNSAVER XSHAPE XSYNC XTEST)
        set(SDL_X11_${extension} OFF CACHE BOOL "" FORCE)
    endforeach()
endif()
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
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # SDL's own X11 sources, built without XRANDR and XSYNC (above), leave a
    # label and a variable unused (SDL_x11toolkit.c, SDL_x11window.c): its
    # warnings, those two named off for its target alone.
    target_compile_options(SDL3-static PRIVATE -Wno-unused-label -Wno-unused-variable)
endif()

# The RTX 3090 Ti's: OptiX's headers, for the CUDA backend's rays.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
# ---------------------------------------------------------------------------
# OptiX 9.1.0: NVIDIA's header-only distribution of the API, whose runtime
# is the driver's (cmake/toolchain.json pins it). Pinned by commit; fetched
# here and never committed, since NVIDIA's licence allows redistributing
# binaries alone (docs/research/2026-10-19-cuda-backend.md). Its host half
# reaches the driver through a function table (optix_stubs.h), which one
# file defines (src/cuda/device/optix_context.cpp), and its device half is
# included by the programs alone.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    optix_dev
    GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
    GIT_TAG        f1f6dd803f3159992d248178f6e09421c6eb8b6d  # OptiX 9.1.0
)
FetchContent_MakeAvailable(optix_dev)
add_library(optix_headers INTERFACE)
target_include_directories(optix_headers SYSTEM INTERFACE "${optix_dev_SOURCE_DIR}/include")
set(SERENITY_OPTIX_INCLUDE_DIR "${optix_dev_SOURCE_DIR}/include")
endif()

# toml++: reading frame graph and scene files (src/core/frame/graph_file.cpp and
# src/core/scene/scene.cpp only). v3.4.0.
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        30172438cee64926dc41fdd9c11fb3ba5b2ba9de  # v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

# stb_image_write: PNG output (src/core/output/png.cpp only; its
# implementation compiled in src/core/output/stb_image_write.cpp). v1.16.
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
    src/core/denoise/json.cpp
    src/core/denoise/network.cpp
    src/core/denoise/unet.cpp
    src/core/materials/coated.cpp
    src/core/film/linear_image.cpp
    src/core/frame/frame_constants.cpp
    src/core/frame/graph_file.cpp
    src/core/frame/history.cpp
    src/core/frame/refusals.cpp
    src/core/frame/schedule.cpp
    src/core/measurement/error.cpp
    src/core/measurement/frame_times.cpp
    src/core/measurement/reference.cpp
    src/core/output/image_format.cpp
    src/core/output/pfm.cpp
    src/core/output/png.cpp
    src/core/output/stb_image_write.cpp
    src/core/passes/depth_of_field.cpp
    src/core/passes/restir_pt_arrays.cpp
    src/core/scene/packed_range.cpp
    src/core/scene/scene.cpp
    src/core/scene/scene_arrays.cpp
    src/core/scene/swarm.cpp
    src/core/presentation/frame_source.cpp
    src/core/sampling/pairing.cpp
    src/core/shapes/shapes.cpp
    src/core/shapes/structure_plan.cpp
)
target_include_directories(serenity_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(serenity_core PRIVATE tomlplusplus::tomlplusplus stb_image_write)
target_compile_options(serenity_core PRIVATE ${SERENITY_SOURCE_WARNINGS})

if(APPLE)
# ---------------------------------------------------------------------------
# Metal backend: the only code that uses Metal's host API. Its shaders are
# compiled into it (cmake/MetalLibrary.cmake), as serenity::metallib::shaders.
# ---------------------------------------------------------------------------
add_library(serenity_metal STATIC
    src/metal/device/metal_cpp.cpp
    src/metal/device/device.cpp
    src/metal/device/display_link.mm
    src/metal/device/library.cpp
    src/metal/device/offscreen.cpp
    src/metal/device/presenter.cpp
    src/metal/device/frame_array.cpp
    src/metal/device/frame_sequence.cpp
    src/metal/device/readback.cpp
    src/metal/device/resident.cpp
    src/metal/device/static_arrays.cpp
    src/metal/device/submission.cpp
    src/metal/device/support.cpp
    src/metal/acceleration/scene_acceleration.cpp
    src/metal/scene/light_glows.cpp
    src/metal/scene/scene_buffers.cpp
    src/metal/scene/shape_transforms.cpp
    src/metal/film/failure_count.cpp
    src/metal/frame/accumulation.cpp
    src/metal/frame/checkerboard_images.cpp
    src/metal/frame/depth_of_field_images.cpp
    src/metal/frame/denoise_images.cpp
    src/metal/frame/display_frames.cpp
    src/metal/frame/feature_images.cpp
    src/metal/frame/frame_images.cpp
    src/metal/frame/mnee_list.cpp
    src/metal/frame/offline_frames.cpp
    src/metal/frame/renderer.cpp
    src/metal/passes/checkerboard_fill/checkerboard_fill.cpp
    src/metal/passes/denoise/denoise.cpp
    src/metal/passes/depth_of_field/depth_of_field.cpp
    src/metal/passes/restir_pt/shift_replay.cpp
    src/metal/passes/features/features.cpp
    src/metal/passes/mnee/mnee.cpp
    src/metal/passes/path/path.cpp
    src/metal/passes/restir_pt_duplication/restir_pt_duplication.cpp
    src/metal/passes/restir_pt_initial/restir_pt_initial.cpp
    src/metal/passes/restir_pt_shade/restir_pt_shade.cpp
    src/metal/passes/restir_pt_spatial/restir_pt_spatial.cpp
    src/metal/passes/restir_pt_temporal/restir_pt_temporal.cpp
    src/metal/passes/test_pattern/test_pattern.cpp
    src/metal/passes/tone_map/tone_map.cpp
)
target_link_libraries(serenity_metal PUBLIC serenity_core metal_cpp)
target_compile_options(serenity_metal PRIVATE ${SERENITY_SOURCE_WARNINGS})
# The display link's Objective-C objects are held under ARC, released by
# what holds them (display_link.mm).
set_source_files_properties(src/metal/device/display_link.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
serenity_add_metallib(
    TARGET serenity_metal NAME shaders
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/test_pattern/test_pattern.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/path/path.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/mnee/mnee.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/tone_map/tone_map.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/restir_pt_initial/restir_pt_initial.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/restir_pt_temporal/restir_pt_temporal.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/restir_pt_spatial/restir_pt_spatial.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/restir_pt_shade/restir_pt_shade.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/restir_pt_duplication/restir_pt_duplication.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/features/features.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/denoise/denoise.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/checkerboard_fill/checkerboard_fill.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/passes/depth_of_field/depth_of_field.metal"
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")

# ---------------------------------------------------------------------------
# The two ways to run a frame. Each parses its command line in a library of
# its own, so the tests can reach it without a window or a GPU; the window's
# clock, which needs neither, is in its library too.
# ---------------------------------------------------------------------------
endif()

add_library(serenity_app_options STATIC src/app/options.cpp src/app/clock.cpp)
target_link_libraries(serenity_app_options PUBLIC serenity_core)
target_compile_options(serenity_app_options PRIVATE ${SERENITY_SOURCE_WARNINGS})

# The window, on each platform with its backend: Metal on macOS, CUDA on
# Linux, which presents through Vulkan (docs/research/2026-10-20-cuda-window.md).
add_executable(serenity src/app/main.cpp src/app/window.cpp src/app/window_frames.cpp)
target_link_libraries(serenity PRIVATE serenity_app_options SDL3::SDL3-static)
target_compile_options(serenity PRIVATE ${SERENITY_SOURCE_WARNINGS})
if(APPLE)
    target_link_libraries(serenity PRIVATE serenity_metal)
else()
    target_link_libraries(serenity PRIVATE serenity_cuda)
endif()

# ---------------------------------------------------------------------------
# CUDA backend, on Linux: the only code that uses CUDA's host API. Its kernels
# and its OptiX programs are compiled into it (cmake/CudaModules.cmake), one
# module a pass, as serenity::cuda_modules::<pass> and
# serenity::optix_modules::<pass>. Built a phase at a time
# (docs/research/2026-10-19-cuda-backend.md); its renderer refuses the
# passes it does not build yet.
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    include(CudaModules)
    add_library(serenity_cuda STATIC
        src/cuda/acceleration/mesh_levels.cpp
        src/cuda/acceleration/scene_acceleration.cpp
        src/cuda/acceleration/shape_meshes.cpp
        src/cuda/device/buffer.cpp
        src/cuda/device/device.cpp
        src/cuda/device/error.cpp
        src/cuda/device/frame_sequence.cpp
        src/cuda/device/frame_upload.cpp
        src/cuda/device/images.cpp
        src/cuda/device/module.cpp
        src/cuda/device/offscreen.cpp
        src/cuda/device/optix_context.cpp
        src/cuda/device/pipeline.cpp
        src/cuda/device/submission.cpp
        src/cuda/film/failure_count.cpp
        src/cuda/frame/accumulation.cpp
        src/cuda/frame/checkerboard_images.cpp
        src/cuda/frame/denoise_images.cpp
        src/cuda/frame/depth_of_field_images.cpp
        src/cuda/frame/feature_images.cpp
        src/cuda/frame/frame_images.cpp
        src/cuda/frame/mnee_list.cpp
        src/cuda/frame/offline_frames.cpp
        src/cuda/frame/path_reuse_images.cpp
        src/cuda/frame/renderer.cpp
        src/cuda/frame/swapchain_frames.cpp
        src/cuda/frame/traced_program.cpp
        src/cuda/passes/checkerboard_fill/checkerboard_fill.cpp
        src/cuda/passes/denoise/denoise.cpp
        src/cuda/passes/depth_of_field/depth_of_field.cpp
        src/cuda/passes/features/features.cpp
        src/cuda/passes/mnee/mnee.cpp
        src/cuda/passes/path/path.cpp
        src/cuda/passes/restir_pt/shift_replay.cpp
        src/cuda/passes/restir_pt_duplication/restir_pt_duplication.cpp
        src/cuda/passes/restir_pt_initial/restir_pt_initial.cpp
        src/cuda/passes/restir_pt_shade/restir_pt_shade.cpp
        src/cuda/passes/restir_pt_spatial/restir_pt_spatial.cpp
        src/cuda/passes/restir_pt_temporal/restir_pt_temporal.cpp
        src/cuda/passes/test_pattern/test_pattern.cpp
        src/cuda/passes/tone_map/tone_map.cpp
        src/cuda/scene/light_glows.cpp
        src/cuda/scene/scene_buffers.cpp
        src/cuda/scene/shape_transforms.cpp
    )
    # Vulkan, for presenting the window's frames (src/cuda/present/): Ubuntu's
    # loader and headers, pinned (cmake/toolchain.json, vulkan).
    find_package(Vulkan 1.3 REQUIRED)
    target_sources(serenity_cuda PRIVATE
        src/cuda/present/presenter.cpp
        src/cuda/present/swapchain_choice.cpp)
    target_link_libraries(serenity_cuda PUBLIC serenity_core CUDA::cuda_driver optix_headers Vulkan::Vulkan ${CMAKE_DL_LIBS})
    target_compile_options(serenity_cuda PRIVATE ${SERENITY_SOURCE_WARNINGS})
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME test_pattern
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/test_pattern/test_pattern.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME tone_map
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/tone_map/tone_map.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_optix_module(
        TARGET serenity_cuda NAME path
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/path/path.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_optix_module(
        TARGET serenity_cuda NAME restir_pt_initial
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_initial/restir_pt_initial.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_optix_module(
        TARGET serenity_cuda NAME mnee
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/mnee/mnee.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_optix_module(
        TARGET serenity_cuda NAME restir_pt_temporal
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_temporal/restir_pt_temporal.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_optix_module(
        TARGET serenity_cuda NAME restir_pt_spatial
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_spatial/restir_pt_spatial.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_optix_module(
        TARGET serenity_cuda NAME features
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/features/features.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME restir_pt_initial_kernels
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_initial/restir_pt_initial_kernels.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME mnee_kernels
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/mnee/mnee_kernels.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME restir_pt_temporal_kernels
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_temporal/restir_pt_temporal_kernels.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME restir_pt_spatial_kernels
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_spatial/restir_pt_spatial_kernels.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME restir_pt_shade
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_shade/restir_pt_shade.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME restir_pt_duplication
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/restir_pt_duplication/restir_pt_duplication.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME checkerboard_fill
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/checkerboard_fill/checkerboard_fill.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME denoise
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/denoise/denoise.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    serenity_add_cuda_module(
        TARGET serenity_cuda NAME depth_of_field
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/passes/depth_of_field/depth_of_field.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

add_library(serenity_headless_options STATIC src/headless/options.cpp src/headless/times.cpp)
target_link_libraries(serenity_headless_options PUBLIC serenity_core)
target_compile_options(serenity_headless_options PRIVATE ${SERENITY_SOURCE_WARNINGS})

# The headless renderer names no backend: it is linked with its platform's,
# whose library defines the offline renderer's factory
# (core/presentation/offline_renderer.h).
add_executable(serenity-headless src/headless/main.cpp src/headless/run.cpp)
if(APPLE)
    target_link_libraries(serenity-headless PRIVATE serenity_headless_options serenity_metal)
else()
    target_link_libraries(serenity-headless PRIVATE serenity_headless_options serenity_cuda)
endif()
target_compile_options(serenity-headless PRIVATE ${SERENITY_SOURCE_WARNINGS})

# The measuring program (src/measure/options.h): the core alone, no Metal,
# so a reference is made and judged on any machine the core builds on. Its
# command line and its commands in a library of its own, as the two above,
# for the tests.
add_library(serenity_measure_options STATIC src/measure/options.cpp src/measure/commands.cpp)
target_link_libraries(serenity_measure_options PUBLIC serenity_core)
target_compile_options(serenity_measure_options PRIVATE ${SERENITY_SOURCE_WARNINGS})

add_executable(serenity-measure src/measure/main.cpp)
target_link_libraries(serenity-measure PRIVATE serenity_measure_options)
target_compile_options(serenity-measure PRIVATE ${SERENITY_SOURCE_WARNINGS})

enable_testing()
add_subdirectory(tests)
if(APPLE)
    add_subdirectory(tests/gpu)
else()
    add_subdirectory(tests/cuda)
endif()
