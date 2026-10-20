# CUDA tests, as the backend is built a phase at a time
# (docs/research/2026-10-19-cuda-backend.md): unit tests of its own logic,
# which run no GPU (the submission's bookkeeping);
# on the RTX 3090 Ti, the probes, which run each family's device code
# against the core's, and one integration test, the headless program
# rendering as its user runs it. With no CUDA device, or one the kernels are
# not built for, those fail, never skip: a GPU test that passes because it
# ran nothing would claim what was never checked.

# Unit tests of the backend's own logic, apart from the driver: no GPU.
add_executable(serenity_cuda_unit_tests
    frame_sequence_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_cuda_unit_tests PRIVATE serenity_cuda serenity_test_support)
add_test(NAME serenity_cuda_unit_tests COMMAND serenity_cuda_unit_tests)

# On the GPU: the probes, each family's device code against the core's
# computations, one source with Metal's (tests/gpu/<family>_test.cpp, which
# include "probe_runner.h", CUDA's from support/ here); and the integration
# test, the headless program as its user runs it. The probes that trace are
# ray generation programs, run by the tracing harness (support/
# traced_probe.h); the scene's and the frames' tests share Metal's whole-
# frame rig's name (support/accumulating_rig.h); the denoised pipeline's
# and the features' share the frame rig's, the hand-off rig's and the
# denoiser runner's (support/frame_rig.h, hand_off_rig.h,
# denoise_runner.h). ReSTIR PT's probe test
# (tests/gpu/restir_pt_test.cpp) dispatches Metal's kernels itself, and
# stays Metal's until the test audit makes it one source.
add_executable(serenity_cuda_tests
    headless_test.cpp
    ../gpu/acceleration_test.cpp
    ../gpu/bsdf_test.cpp
    ../gpu/camera_test.cpp
    ../gpu/denoise_test.cpp
    ../gpu/encoding_test.cpp
    ../gpu/features_test.cpp
    ../gpu/film_test.cpp
    ../gpu/leave_test.cpp
    ../gpu/light_test.cpp
    ../gpu/mnee_test.cpp
    ../gpu/path_test.cpp
    ../gpu/sampler_test.cpp
    ../gpu/scene_block_test.cpp
    ../gpu/shape_test.cpp
    ../gpu/texture_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_cuda_tests PRIVATE serenity_cuda serenity_test_support)
target_include_directories(serenity_cuda_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/support"
                                                      "${CMAKE_CURRENT_SOURCE_DIR}")
# The probe kernels, one module (kernels/probes.cu), their records in the
# headers they share with Metal's (tests/gpu/kernels/).
serenity_add_cuda_module(
    TARGET serenity_cuda_tests NAME probes
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/probes.cu"
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/tests")
# The probes that trace, one OptiX module (kernels/traced_probes.cu).
serenity_add_optix_module(
    TARGET serenity_cuda_tests NAME traced_probes
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/traced_probes.cu"
    INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/tests"
                 "${CMAKE_CURRENT_SOURCE_DIR}")
# stb_image, to decode the PNG the headless program writes; the same pinned
# stb as the writer's (CMakeLists.txt), test-only.
target_include_directories(serenity_cuda_tests SYSTEM PRIVATE "${stb_SOURCE_DIR}")
# It runs serenity-headless as its user does (tests/support/headless_program.h.in).
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/generated/headless_program.h"
     INPUT "${PROJECT_SOURCE_DIR}/tests/support/headless_program.h.in")
add_dependencies(serenity_cuda_tests serenity-headless)
add_test(NAME serenity_cuda_tests COMMAND serenity_cuda_tests)
