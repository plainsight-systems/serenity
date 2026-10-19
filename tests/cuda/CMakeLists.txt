# CUDA tests: the CUDA backend, run on this machine's GPU, the RTX 3090 Ti.
# With no CUDA device, or one the kernels are not built for, they fail,
# never skip: a GPU test that passes because it ran nothing would claim what
# was never checked. The Metal tests' counterparts (tests/gpu), as the
# backend is built a phase at a time (docs/research/2026-10-19-cuda-
# backend.md): the device and the machine it names, the submission's
# protocol, the test pattern against the CPU's, and the headless program
# rendering it as its user runs it.

add_executable(serenity_cuda_tests
    device_test.cpp
    submission_test.cpp
    test_pattern_test.cpp
    headless_test.cpp
    ../main_test.cpp
)
target_link_libraries(serenity_cuda_tests PRIVATE serenity_cuda serenity_test_support)
# stb_image, to decode the PNG the headless program writes; the same pinned
# stb as the writer's (CMakeLists.txt), test-only.
target_include_directories(serenity_cuda_tests SYSTEM PRIVATE "${stb_SOURCE_DIR}")
# headless_test runs serenity-headless as its user does
# (tests/support/headless_program.h.in).
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/generated/headless_program.h"
     INPUT "${PROJECT_SOURCE_DIR}/tests/support/headless_program.h.in")
add_dependencies(serenity_cuda_tests serenity-headless)
add_test(NAME serenity_cuda_tests COMMAND serenity_cuda_tests)
