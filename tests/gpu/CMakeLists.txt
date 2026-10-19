# Metal's tests, by the project's rule that unit tests test units of code
# and tests that run the GPU are units of GPU code or a handful of
# integration tests (CLAUDE.md):
#
#   - serenity_metal_unit_tests: the backend's own host logic, apart from
#     Metal, with no GPU: the submission's bookkeeping, the image size rule.
#   - serenity_gpu_tests, on this machine's GPU (with no Metal device, or
#     one without ray tracing, they fail, never skip: a GPU test that passes
#     because it ran nothing would claim what was never checked):
#       - units of GPU code: the shader functions through the probe kernels
#         below, each against the CPU's computation or a closed form; the
#         tone-map pass on images written for it;
#       - our Metal wrappers' refusals and lifetimes that need the device
#         (the library, the presenter, the display link's thread, the memory
#         owners' bounds), and the platform's behaviour the design relies on
#         (the argument table's snapshot, the largest image);
#       - a handful of integration tests, frames rendered end to end, each
#         for an invariant no unit shows: the path tracer's mean on closed
#         forms (it renders the training targets and the references),
#         ReSTIR PT keeping it, MNEE counting the caustic once, the Metal
#         denoiser matching PyTorch, its two halves' hand-off, determinism,
#         frames in flight each reading their own constants, nothing
#         allocated recording a frame, the GPU waited for before what it uses
#         is freed, the features beside the light changing it in no bit, and
#         the headless program's PFMs being the accumulated image.

# Where the headless renderer is (support/headless_program.h.in), beside
# the tests' other generated headers (tests/CMakeLists.txt).
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/generated/headless_program.h"
     INPUT "${PROJECT_SOURCE_DIR}/tests/support/headless_program.h.in")

# Unit tests of the Metal backend's own logic, apart from Metal: no GPU.
add_executable(serenity_metal_unit_tests
    frame_sequence_test.cpp
    texture_size_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_metal_unit_tests PRIVATE serenity_metal serenity_test_support)
add_test(NAME serenity_metal_unit_tests COMMAND serenity_metal_unit_tests)

add_executable(serenity_gpu_tests
    library_test.cpp
    frame_test.cpp
    argument_table_test.cpp
    light_test.cpp
    bsdf_test.cpp
    sampler_test.cpp
    path_test.cpp
    mnee_test.cpp
    restir_pt_test.cpp
    encoding_test.cpp
    features_test.cpp
    film_test.cpp
    acceleration_test.cpp
    leave_test.cpp
    tone_map_test.cpp
    texture_test.cpp
    camera_test.cpp
    shape_test.cpp
    scene_block_test.cpp
    allocation_test.cpp
    lifetime_test.cpp
    frame_errors_test.cpp
    headless_pfm_test.cpp
    denoise_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_gpu_tests PRIVATE serenity_metal serenity_test_support)
# headless_pfm_test runs serenity-headless as its user does
# (tests/support/programs.h.in).
add_dependencies(serenity_gpu_tests serenity-headless)

# The probe kernels, each running a shader half on its own for its test
# (each kernel's records in its own header, kernels/<kernel>.h, shared with
# its test alone); the smallest kernel, smoke, that the library and
# argument-table tests run; and delay, which the hand-off test holds a
# queue back with.
serenity_add_metallib(
    TARGET serenity_gpu_tests NAME probes
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/smoke.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/delay.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/light_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/bsdf_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/sampler_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/film_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/trace_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/leave_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/texture_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/camera_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/shape_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/scene_block_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/encoding_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/path_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/restir_pt_probe.metal"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mnee_probe.metal"
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src")

# Under Metal's API and shader validation, in ctest as by hand (P.12): an
# out-of-bounds read, a use of freed memory or a misuse of the API is then a
# failure the run reports, not garbage a check may or may not see.
add_test(NAME serenity_gpu_tests COMMAND serenity_gpu_tests)
set_tests_properties(serenity_gpu_tests PROPERTIES
    ENVIRONMENT "MTL_DEBUG_LAYER=1;MTL_SHADER_VALIDATION=1;MTL_DEBUG_LAYER_ERROR_MODE=assert")

# The denoiser against PyTorch on a real network's frames, beside the checkout
# (denoise_check.cpp says how): a tool, not a ctest test.
add_executable(denoise_check denoise_check.cpp)
target_link_libraries(denoise_check PRIVATE serenity_metal serenity_test_support)
# The shaders' library header, generated beside the probes' (cmake/MetalLibrary.cmake).
target_include_directories(denoise_check PRIVATE "${CMAKE_BINARY_DIR}/generated")
