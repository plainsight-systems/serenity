# CUDA tests, as the backend is built a phase at a time
# (docs/research/2026-10-19-cuda-backend.md): unit tests of its own logic,
# which run no GPU (the submission's bookkeeping, which passes it builds),
# and one integration test on the RTX 3090 Ti, the headless program
# rendering as its user runs it. With no CUDA device, or one the kernels are
# not built for, that one fails, never skips: a GPU test that passes because
# it ran nothing would claim what was never checked.

# Unit tests of the backend's own logic, apart from the driver: no GPU.
add_executable(serenity_cuda_unit_tests
    frame_sequence_test.cpp
    unbuilt_pass_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_cuda_unit_tests PRIVATE serenity_cuda serenity_test_support)
add_test(NAME serenity_cuda_unit_tests COMMAND serenity_cuda_unit_tests)

# The integration test, on the GPU: the headless program as its user runs it.
add_executable(serenity_cuda_tests
    headless_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_cuda_tests PRIVATE serenity_cuda serenity_test_support)
# stb_image, to decode the PNG the headless program writes; the same pinned
# stb as the writer's (CMakeLists.txt), test-only.
target_include_directories(serenity_cuda_tests SYSTEM PRIVATE "${stb_SOURCE_DIR}")
# It runs serenity-headless as its user does (tests/support/headless_program.h.in).
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/generated/headless_program.h"
     INPUT "${PROJECT_SOURCE_DIR}/tests/support/headless_program.h.in")
add_dependencies(serenity_cuda_tests serenity-headless)
add_test(NAME serenity_cuda_tests COMMAND serenity_cuda_tests)
