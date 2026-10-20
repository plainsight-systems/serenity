# CUDA tests, as the backend is built a phase at a time
# (docs/research/2026-10-19-cuda-backend.md): unit tests of its own logic,
# which run no GPU (the submission's bookkeeping, which passes it builds);
# on the RTX 3090 Ti, the probes, which run each family's device code
# against the core's, and one integration test, the headless program
# rendering as its user runs it. With no CUDA device, or one the kernels are
# not built for, those fail, never skip: a GPU test that passes because it
# ran nothing would claim what was never checked.

# Unit tests of the backend's own logic, apart from the driver: no GPU.
add_executable(serenity_cuda_unit_tests
    frame_sequence_test.cpp
    unbuilt_pass_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_cuda_unit_tests PRIVATE serenity_cuda serenity_test_support)
add_test(NAME serenity_cuda_unit_tests COMMAND serenity_cuda_unit_tests)

# On the GPU: the probes, each family's device code against the core's
# computations, one source with Metal's (tests/gpu/<family>_test.cpp, which
# include "probe_runner.h", CUDA's from support/ here); and the integration
# test, the headless program as its user runs it. The probe tests of the
# families that trace (acceleration, leaving a surface, the path loop) run
# OptiX launches, and join with Phase 4's tracing probes.
add_executable(serenity_cuda_tests
    headless_test.cpp
    ../gpu/bsdf_test.cpp
    ../gpu/camera_test.cpp
    ../gpu/encoding_test.cpp
    ../gpu/film_test.cpp
    ../gpu/light_test.cpp
    ../gpu/sampler_test.cpp
    ../gpu/shape_test.cpp
    ../gpu/texture_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_cuda_tests PRIVATE serenity_cuda serenity_test_support)
target_include_directories(serenity_cuda_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/support")
# The probe kernels, one module (kernels/probes.cu), their records in the
# headers they share with Metal's (tests/gpu/kernels/).
serenity_add_cuda_module(
    TARGET serenity_cuda_tests NAME probes
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/probes.cu"
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/tests")
# stb_image, to decode the PNG the headless program writes; the same pinned
# stb as the writer's (CMakeLists.txt), test-only.
target_include_directories(serenity_cuda_tests SYSTEM PRIVATE "${stb_SOURCE_DIR}")
# It runs serenity-headless as its user does (tests/support/headless_program.h.in).
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/generated/headless_program.h"
     INPUT "${PROJECT_SOURCE_DIR}/tests/support/headless_program.h.in")
add_dependencies(serenity_cuda_tests serenity-headless)
add_test(NAME serenity_cuda_tests COMMAND serenity_cuda_tests)
