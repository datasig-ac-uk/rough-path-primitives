include(FetchContent)

find_package(benchmark CONFIG QUIET)

if (NOT benchmark_FOUND)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.9.4
            GIT_SHALLOW TRUE
            FIND_PACKAGE_ARGS CONFIG
    )
    FetchContent_MakeAvailable(benchmark)
endif ()

add_subdirectory(cpu)

if (RPP_ENABLE_CUDA)
    #    add_subdirectory(gpu)
endif ()
