# Metal libraries, compiled when the project builds and compiled into the
# executable that uses them.
#
#   serenity_add_metallib(
#       TARGET       <target that uses the library>
#       NAME         <C++ identifier for the library's bytes>
#       SOURCES      <.metal files>...
#       INCLUDE_DIRS <directories the shaders include from>...
#       [MATH fast])
#
# Each source is compiled to AIR with SERENITY_METAL_FLAGS, the AIR is linked
# into one .metallib, and its bytes become
#
#   serenity::metallib::<NAME>   (inline constexpr unsigned char[])
#
# in ${CMAKE_BINARY_DIR}/generated/serenity/metallib/<NAME>.h, which TARGET
# can include as "serenity/metallib/<NAME>.h" and hand to metal::Library.
# Compiled in, not loaded from a path: what a shader computes is fixed by the
# source revision, and nothing at run time depends on where the build put it.
#
# The flags are pinned here and nowhere else (core/portable_math.h says why
# each is needed):
#
#   -std=metal3.2                  the language version, explicit rather than
#                                  the compiler's default
#   -fmetal-math-mode=safe         Metal's default is fast, which reassociates
#   -fmetal-math-fp32-functions=precise
#                                  and approximates division and sqrt
#   -ffp-contract=off              a * b + c is not fused; fma() is
#   -mmacosx-version-min           the deployment target the C++ builds for
#   -Werror                        a shader warning fails the build
#
# MATH fast compiles with Metal's fast math instead. It exists for a kernel
# that opts out of bit-equality with the CPU as a labelled, measured
# optimization, and for the test that shows why the default is refused.
#
# Dependencies on included headers are tracked through the compiler's depfile,
# so changing a shared header such as core/portable_math.h rebuilds every
# library that includes it.

set(SERENITY_METAL_STD "-std=metal3.2")
set(SERENITY_METAL_MATH_SAFE
    -fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off)
set(SERENITY_METAL_MATH_FAST
    -fmetal-math-mode=fast -fmetal-math-fp32-functions=fast -ffp-contract=fast)

function(serenity_add_metallib)
    cmake_parse_arguments(ARG "" "TARGET;NAME;MATH" "SOURCES;INCLUDE_DIRS" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_NAME OR NOT ARG_SOURCES)
        message(FATAL_ERROR "serenity_add_metallib: TARGET, NAME and SOURCES are required")
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        message(FATAL_ERROR "serenity_add_metallib: CMAKE_OSX_DEPLOYMENT_TARGET is not set (CMakePresets.json sets it)")
    endif()

    if(NOT ARG_MATH)
        set(math ${SERENITY_METAL_MATH_SAFE})
    elseif(ARG_MATH STREQUAL "fast")
        set(math ${SERENITY_METAL_MATH_FAST})
    else()
        message(FATAL_ERROR "serenity_add_metallib: MATH is 'fast' or absent, not '${ARG_MATH}'")
    endif()

    set(work "${CMAKE_BINARY_DIR}/metallib/${ARG_NAME}")
    file(MAKE_DIRECTORY "${work}" "${CMAKE_BINARY_DIR}/generated/serenity/metallib")
    set(includes "")
    foreach(dir IN LISTS ARG_INCLUDE_DIRS)
        list(APPEND includes "-I${dir}")
    endforeach()

    set(airs "")
    foreach(source IN LISTS ARG_SOURCES)
        get_filename_component(stem "${source}" NAME_WE)
        set(air "${work}/${stem}.air")
        set(dep "${work}/${stem}.d")
        add_custom_command(
            OUTPUT "${air}"
            COMMAND xcrun -sdk macosx metal
                    ${SERENITY_METAL_STD} ${math}
                    -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
                    -Werror ${includes}
                    -MMD -MF "${dep}"
                    -c "${source}" -o "${air}"
            DEPENDS "${source}"
            DEPFILE "${dep}"
            COMMENT "Metal: compiling ${stem}.metal for ${ARG_NAME}"
            VERBATIM)
        list(APPEND airs "${air}")
    endforeach()

    set(metallib "${work}/${ARG_NAME}.metallib")
    add_custom_command(
        OUTPUT "${metallib}"
        COMMAND xcrun -sdk macosx metallib ${airs} -o "${metallib}"
        DEPENDS ${airs}
        COMMENT "Metal: linking ${ARG_NAME}.metallib"
        VERBATIM)

    set(header "${CMAKE_BINARY_DIR}/generated/serenity/metallib/${ARG_NAME}.h")
    add_custom_command(
        OUTPUT "${header}"
        COMMAND "${CMAKE_COMMAND}"
                -DINPUT=${metallib} -DOUTPUT=${header} -DNAME=${ARG_NAME}
                -P "${SERENITY_CMAKE_DIR}/embed_metallib_script.cmake"
        DEPENDS "${metallib}" "${SERENITY_CMAKE_DIR}/embed_metallib_script.cmake"
        COMMENT "Metal: embedding ${ARG_NAME}.metallib"
        VERBATIM)

    target_sources(${ARG_TARGET} PRIVATE "${header}")
    target_include_directories(${ARG_TARGET} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()
