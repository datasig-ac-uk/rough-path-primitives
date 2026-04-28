cmake_minimum_required(VERSION 4.2)

include(GNUInstallDirs)

project(RoughPathPrimitives
        VERSION 0.0.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)



option(RPP_ENABLE_CUDA "Enable cuda support" OFF)
option(RPP_ENABLE_TESTS "Enable building tests" ${PROJECT_IS_TOP_LEVEL})
option(RPP_ENABLE_BENCHMARKS "Enable building benchmarks" ${PROJECT_IS_TOP_LEVEL})




add_library(RPP INTERFACE)
add_library(RPP::RPP ALIAS RPP)

target_include_directories(RPP INTERFACE
        $<BUILD_INTERFACE:include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


target_sources(RPP INTERFACE FILE_SET HEADERS BASE_DIRS include FILES
        include/rpp/config/compiler.h
        include/rpp/config/target.h
        include/rpp/dense/batch.hpp
        include/rpp/dense/views.hpp
        include/rpp/architecture.hpp
        include/rpp/basis.hpp
        include/rpp/config.h
        include/rpp/integer_maths.hpp
        include/rpp/operations.hpp
        include/rpp/utility.hpp
)


set(RPP_GPU_SOURCES
        rpp/gpu/ops/block/detail/ft_adjoint_multiply.hpp
        rpp/gpu/ops/block/detail/ft_multiply.hpp
        rpp/gpu/ops/block/detail/st_multiply.hpp
        rpp/gpu/ops/block/ft_adj_lmul.hpp
        rpp/gpu/ops/block/ft_adj_rmul.hpp
        rpp/gpu/ops/block/ft_exp.hpp
        rpp/gpu/ops/block/ft_fma.hpp
        rpp/gpu/ops/block/ft_fmexp.hpp
        rpp/gpu/ops/block/ft_inplace_fma.hpp
        rpp/gpu/ops/block/ft_inplace_mul.hpp
        rpp/gpu/ops/block/ft_log.hpp
        rpp/gpu/ops/block/ft_mul.hpp
        rpp/gpu/ops/block/st_adj_mul.hpp
        rpp/gpu/ops/block/st_fma.hpp
        rpp/gpu/ops/block/st_inplace_fma.hpp
        rpp/gpu/ops/block/st_mul.hpp
        rpp/gpu/ops/block/tensor_add_identity.hpp
        rpp/gpu/ops/block/tensor_antipode.hpp
        rpp/gpu/ops/block/tensor_reflect.hpp
        rpp/gpu/ops/block/tensor_set_identity.hpp
        rpp/gpu/ops/block/vector_assign.hpp
        rpp/gpu/ops/block/vector_set_constant.hpp
        rpp/gpu/ops/block/vector_set_zero.hpp
        rpp/gpu/architecture.hpp
        rpp/gpu/kernel.hpp
        rpp/gpu/strategies.hpp
)

if (RPP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit CONFIG REQUIRED)

    target_sources(RPP INTERFACE FILE_SET HEADERS BASE_DIRS include FILES
        ${RPP_GPU_SOURCES}
    )
    set_source_files_properties(${RPP_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()



