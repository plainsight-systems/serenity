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
    sampler_test.cpp
    path_test.cpp
    film_test.cpp
    acceleration_test.cpp
    motion_test.cpp
    history_test.cpp
    tone_map_test.cpp
    frame_images_test.cpp
    texture_test.cpp
    camera_test.cpp
    shape_test.cpp
    scene_block_test.cpp
    allocation_test.cpp
    lifetime_test.cpp
    frame_errors_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_gpu_tests PRIVATE serenity_metal serenity_test_support)

# The probe kernels, each running a shader half on its own for its test
# (kernels/probes.h), and the smallest kernel, smoke, that the library and
# argument-table tests run.
serenity_add_metallib(
    TARGET serenity_gpu_tests NAME probes
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/smoke.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/light_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/bsdf_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sampler_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/film_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/trace_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/texture_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/camera_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/shape_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/scene_block_probe.metal"
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")

# Under Metal's API and shader validation, in ctest as by hand (P.12): an
# out-of-bounds read, a use of freed memory or a misuse of the API is then a
# failure the run reports, not garbage a check may or may not see.
add_test(NAME serenity_gpu_tests COMMAND serenity_gpu_tests)
set_tests_properties(serenity_gpu_tests PROPERTIES
    ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1;MTL_DEBUG_LAYER_ERROR_MODE=assert")
