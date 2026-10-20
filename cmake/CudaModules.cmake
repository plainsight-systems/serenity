# CUDA modules and OptiX programs, compiled when the project builds and
# compiled into the executable that uses them, as Metal's libraries are
# (MetalLibrary.cmake).
#
#   serenity_add_cuda_module(
#       TARGET       <target that uses the module>
#       NAME         <C++ identifier for the module's bytes>
#       SOURCE       <one .cu file>
#       INCLUDE_DIRS <directories the kernels include from>...)
#
# The source is compiled by nvcc to a fatbin with the flags below, and its
# bytes become
#
#   serenity::cuda_modules::<NAME>   (std::span<const std::byte> over a constexpr std::array)
#
# in ${CMAKE_BINARY_DIR}/generated/serenity/cuda_modules/<NAME>.h, which
# TARGET can include as "serenity/cuda_modules/<NAME>.h" and hand to
# cuda::Module. Compiled in, not loaded from a path: what a kernel computes
# is fixed by the source revision, and nothing at run time depends on where
# the build put it. One module a source: each pass's kernels are their own
# translation unit, no relocatable device code to link between them.
#
# nvcc compiles device code only (docs/research/2026-10-19-cuda-backend.md):
# a `.cu` file holds kernels, and the host code that launches them is C++
# the pinned host compiler builds with the project's warnings. nvcc is
# given that compiler for the host half it still parses.
#
# The flags are pinned here and nowhere else:
#
#   --fatbin -gencode arch=compute_86,code=sm_86
#                         machine code for the RTX 3090 Ti alone (Ampere,
#                         sm_86), no PTX the driver would compile at load:
#                         what runs is what the build made
#                         (cuda/device/device.h refuses any other device)
#   -std=c++20            the project's language
#   -O3                   device code's optimization, named, nvcc's default
#   -ccbin                the pinned host compiler (cmake/toolchain.json)
#   -Werror all-warnings  a warning in a kernel fails the build
#
# No fast math (no --use_fast_math, no -ftz): IEEE division and square root,
# denormals kept, as the CPU computes, which the GPU tests compare with
# (docs/research/2026-10-19-cuda-backend.md, "Math mode"). Dependencies on
# included headers are tracked through nvcc's depfile.

set(SERENITY_CUDA_ARCH "-gencode" "arch=compute_86,code=sm_86")
set(SERENITY_CUDA_STD "-std=c++20")
set(SERENITY_CUDA_OPTIMIZATION "-O3")

function(serenity_add_cuda_module)
    cmake_parse_arguments(ARG "" "TARGET;NAME;SOURCE" "INCLUDE_DIRS" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_NAME OR NOT ARG_SOURCE)
        message(FATAL_ERROR "serenity_add_cuda_module: TARGET, NAME and SOURCE are required")
    endif()
    if(NOT CUDAToolkit_NVCC_EXECUTABLE)
        message(FATAL_ERROR "serenity_add_cuda_module: no nvcc (find_package(CUDAToolkit))")
    endif()

    set(work "${CMAKE_BINARY_DIR}/cuda_modules/${ARG_NAME}")
    file(MAKE_DIRECTORY "${work}" "${CMAKE_BINARY_DIR}/generated/serenity/cuda_modules")
    set(includes "")
    foreach(dir IN LISTS ARG_INCLUDE_DIRS)
        list(APPEND includes "-I${dir}")
    endforeach()

    set(fatbin "${work}/${ARG_NAME}.fatbin")
    set(dep "${work}/${ARG_NAME}.d")
    add_custom_command(
        OUTPUT "${fatbin}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                --fatbin ${SERENITY_CUDA_ARCH} ${SERENITY_CUDA_STD} ${SERENITY_CUDA_OPTIMIZATION}
                -ccbin "${CMAKE_CXX_COMPILER}" -Werror all-warnings ${includes}
                -MD -MF "${dep}"
                "${ARG_SOURCE}" -o "${fatbin}"
        DEPENDS "${ARG_SOURCE}"
        DEPFILE "${dep}"
        COMMENT "CUDA: compiling ${ARG_NAME}"
        VERBATIM)

    set(header "${CMAKE_BINARY_DIR}/generated/serenity/cuda_modules/${ARG_NAME}.h")
    add_custom_command(
        OUTPUT "${header}"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT=${fatbin} -DOUTPUT=${header} -DNAME=${ARG_NAME} -DNAMESPACE=cuda_modules
                -P "${SERENITY_CMAKE_DIR}/embed_bytes_script.cmake"
        DEPENDS "${fatbin}" "${SERENITY_CMAKE_DIR}/embed_bytes_script.cmake"
        COMMENT "CUDA: embedding ${ARG_NAME}"
        VERBATIM)

    target_sources(${ARG_TARGET} PRIVATE "${header}")
    target_include_directories(${ARG_TARGET} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()

# OptiX programs: a pass that traces is an OptiX pipeline, its programs one
# `.cu` file compiled to OptiX-IR (docs/research/2026-10-19-cuda-backend.md,
# "Phase 3's design"), embedded as
#
#   serenity::optix_modules::<NAME>   (std::span<const std::byte>)
#
# in ${CMAKE_BINARY_DIR}/generated/serenity/optix_modules/<NAME>.h, which
# TARGET hands to cuda::Pipeline:
#
#   serenity_add_optix_module(
#       TARGET       <target that uses the module>
#       NAME         <C++ identifier for the module's bytes>
#       SOURCE       <one .cu file of programs>
#       INCLUDE_DIRS <directories the programs include from>...)
#
# The flags, from the OptiX 9.1 Programming Guide's list (6.1, "Program
# input"), pinned here and nowhere else:
#
#   -optix-ir             OptiX's IR, which the guide recommends over PTX
#                         for nvcc (its optimizations of live state); OptiX
#                         compiles it to the machine's code when the module
#                         is made, at start-up
#   -arch=compute_86      the IR for the RTX 3090 Ti's architecture, no
#                         higher than the GPU's, as the guide requires
#   -m64                  64-bit code, the only kind OptiX takes
#   -rdc=true             relocatable device code, as the guide asks, so
#                         nvcc removes no program as unreferenced
#   -std=c++20 -O3 -ccbin -Werror all-warnings, as the kernels'
#   -isystem              OptiX's headers (SERENITY_OPTIX_INCLUDE_DIR, set
#                         where they are fetched), whose warnings are theirs
#
# No --use_fast_math, which the guide recommends for speed and declines for
# precision, the kernels' reason above.

set(SERENITY_OPTIX_ARCH "-arch=compute_86")

function(serenity_add_optix_module)
    cmake_parse_arguments(ARG "" "TARGET;NAME;SOURCE" "INCLUDE_DIRS" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_NAME OR NOT ARG_SOURCE)
        message(FATAL_ERROR "serenity_add_optix_module: TARGET, NAME and SOURCE are required")
    endif()
    if(NOT CUDAToolkit_NVCC_EXECUTABLE)
        message(FATAL_ERROR "serenity_add_optix_module: no nvcc (find_package(CUDAToolkit))")
    endif()
    if(NOT SERENITY_OPTIX_INCLUDE_DIR)
        message(FATAL_ERROR "serenity_add_optix_module: no OptiX headers (SERENITY_OPTIX_INCLUDE_DIR)")
    endif()

    set(work "${CMAKE_BINARY_DIR}/optix_modules/${ARG_NAME}")
    file(MAKE_DIRECTORY "${work}" "${CMAKE_BINARY_DIR}/generated/serenity/optix_modules")
    set(includes "")
    foreach(dir IN LISTS ARG_INCLUDE_DIRS)
        list(APPEND includes "-I${dir}")
    endforeach()

    set(ir "${work}/${ARG_NAME}.optixir")
    set(dep "${work}/${ARG_NAME}.d")
    add_custom_command(
        OUTPUT "${ir}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                -optix-ir ${SERENITY_OPTIX_ARCH} -m64 -rdc=true ${SERENITY_CUDA_STD} ${SERENITY_CUDA_OPTIMIZATION}
                -ccbin "${CMAKE_CXX_COMPILER}" -Werror all-warnings ${includes}
                -isystem "${SERENITY_OPTIX_INCLUDE_DIR}"
                -MD -MF "${dep}"
                "${ARG_SOURCE}" -o "${ir}"
        DEPENDS "${ARG_SOURCE}"
        DEPFILE "${dep}"
        COMMENT "OptiX: compiling ${ARG_NAME}"
        VERBATIM)

    set(header "${CMAKE_BINARY_DIR}/generated/serenity/optix_modules/${ARG_NAME}.h")
    add_custom_command(
        OUTPUT "${header}"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT=${ir} -DOUTPUT=${header} -DNAME=${ARG_NAME} -DNAMESPACE=optix_modules
                -P "${SERENITY_CMAKE_DIR}/embed_bytes_script.cmake"
        DEPENDS "${ir}" "${SERENITY_CMAKE_DIR}/embed_bytes_script.cmake"
        COMMENT "OptiX: embedding ${ARG_NAME}"
        VERBATIM)

    target_sources(${ARG_TARGET} PRIVATE "${header}")
    target_include_directories(${ARG_TARGET} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()
