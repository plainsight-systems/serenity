# Pinned by commit: dependency resolution must be deterministic, and a
# floating ref would make the test build a moving target.
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        1da23a3e8119ec5cce4f9388e91b065e20bf06f5
)
FetchContent_MakeAvailable(doctest)
get_target_property(doctest_includes doctest::doctest INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(doctest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${doctest_includes}")

# The core's tests: no GPU, and no platform API.
add_executable(serenity_tests
    contract_headers_test.cpp
    portable_math_test.cpp
    main_test.cpp
)
target_link_libraries(serenity_tests PRIVATE serenity_core doctest::doctest)
target_compile_options(serenity_tests PRIVATE ${SERENITY_WARNINGS})
add_test(NAME serenity_tests COMMAND serenity_tests)
