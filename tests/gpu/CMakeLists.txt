# GPU tests: the Metal backend, run on this machine's GPU. With no Metal
# device, or one without ray tracing, they fail, never skip: a GPU test that
# passes because it ran nothing would claim what was never checked.
add_executable(serenity_gpu_tests
    device_test.cpp
    library_test.cpp
    frame_test.cpp
    argument_table_test.cpp
    preview_test.cpp
    light_test.cpp
    bsdf_test.cpp
    path_test.cpp
    film_test.cpp
    acceleration_test.cpp
    motion_test.cpp
    history_test.cpp
    tone_map_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_gpu_tests PRIVATE serenity_metal doctest::doctest)
target_compile_options(serenity_gpu_tests PRIVATE ${SERENITY_WARNINGS})
target_compile_definitions(serenity_gpu_tests PRIVATE
    SERENITY_SCENES_DIR="${PROJECT_SOURCE_DIR}/scenes"
    SERENITY_GRAPHS_DIR="${PROJECT_SOURCE_DIR}/graphs")

serenity_add_metallib(
    TARGET serenity_gpu_tests NAME smoke
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/smoke.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/light_cone.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/bsdf_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/film_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/trace_probe.metal"
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")

add_test(NAME serenity_gpu_tests COMMAND serenity_gpu_tests)
