# Metal libraries, compiled when the project builds and compiled into the
# executable that uses them.
#
#   serenity_add_metallib(
#       TARGET       <target that uses the library>
#       NAME         <C++ identifier for the library's bytes>
#       SOURCES      <.metal files>...
#       INCLUDE_DIRS <directories the shaders include from>...)
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
# The flags are pinned here and nowhere else:
#
#   -std=metal4.0                  the language version, explicit rather than
#                                  the compiler's default
#   -mmacosx-version-min           the deployment target the C++ builds for
#   -Werror                        a shader warning fails the build
#
# Math is Metal's default (fast). Dependencies on included headers are
# tracked through the compiler's depfile, so changing a header a shader
# includes rebuilds every library that includes it.

set(SERENITY_METAL_STD "-std=metal4.0")

function(serenity_add_metallib)
    cmake_parse_arguments(ARG "" "TARGET;NAME" "SOURCES;INCLUDE_DIRS" ${ARGN})
    if(NOT ARG_TARGET OR NOT ARG_NAME OR NOT ARG_SOURCES)
        message(FATAL_ERROR "serenity_add_metallib: TARGET, NAME and SOURCES are required")
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        message(FATAL_ERROR "serenity_add_metallib: CMAKE_OSX_DEPLOYMENT_TARGET is not set (CMakePresets.json sets it)")
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
                    ${SERENITY_METAL_STD}
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
