include(FetchContent)


set(RPP_GTEST_GIT_TAG "v1.17.0" CACHE STRING "GoogleTest git tag to use when FetchContent is needed")


set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG ${RPP_GTEST_GIT_TAG}
        GIT_SHALLOW TRUE
        FIND_PACKAGE_ARGS NAMES GTest
        EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)


function(rpp_add_test target)
    add_executable(${target})
    target_sources(${target} PRIVATE ${ARGN})
    target_link_libraries(${target}
            PRIVATE
            RPP::RPP
            GTest::gtest_main
    )

    gtest_discover_tests(${target})
endfunction()


add_subdirectory(rpp)
