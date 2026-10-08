# GPU tests: the Metal backend, run on this machine's GPU. With no Metal
# device, or one without ray tracing, they fail, never skip: a GPU test that
# passes because it ran nothing would claim what was never checked.
add_executable(serenity_gpu_tests
    device_test.cpp
    library_test.cpp
    portable_math_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_gpu_tests PRIVATE serenity_metal doctest::doctest)
target_compile_options(serenity_gpu_tests PRIVATE ${SERENITY_WARNINGS})
target_include_directories(serenity_gpu_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# The probe, built twice from one source: with the project's pinned flags,
# and with Metal's fast math, to show the flags are what make the bits equal.
set(probe_sources "${CMAKE_CURRENT_SOURCE_DIR}/kernels/portable_math_probe.metal")
set(probe_includes "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
serenity_add_metallib(
    TARGET serenity_gpu_tests NAME portable_math_probe
    SOURCES ${probe_sources} INCLUDE_DIRS ${probe_includes})
serenity_add_metallib(
    TARGET serenity_gpu_tests NAME portable_math_probe_fast MATH fast
    SOURCES ${probe_sources} INCLUDE_DIRS ${probe_includes})

add_test(NAME serenity_gpu_tests COMMAND serenity_gpu_tests)
