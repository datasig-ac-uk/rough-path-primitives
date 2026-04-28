cmake_minimum_required(VERSION 4.2)


project(RoughPathPrimitives
        VERSION 0.0.0
        LANGUAGES CXX
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)




option(RPP_ENABLE_CUDA "Enable cuda support" OFF)
option(RPP_ENABLE_TESTS "Enable building tests" ${PROJECT_IS_TOP_LEVEL})
option(RPP_ENABLE_BENCHMARKS "Enable building benchmarks" ${PROJECT_IS_TOP_LEVEL})


include(GNUInstallDirs)


add_library(RPP INTERFACE
        include/rpp/config/compiler.h
        include/rpp/config/target.h
        include/rpp/dense/batch.hpp
        include/rpp/dense/views.hpp
        include/rpp/architecture.hpp
        include/rpp/basis.hpp
        include/rpp/config.h
        include/rpp/operations.hpp
        include/rpp/utility.hpp
        include/rpp/cpu/strategies.hpp
        include/rpp/cpu/ops/single_thread.hpp
        include/rpp/cpu/ops/single_thread/detail/antipode.hpp
        include/rpp/cpu/ops/single_thread/ft_adj_lmul.hpp
        include/rpp/cpu/ops/single_thread/ft_adj_rmul.hpp
        include/rpp/cpu/ops/single_thread/ft_exp.hpp
        include/rpp/cpu/ops/single_thread/ft_fma.hpp
        include/rpp/cpu/ops/single_thread/ft_fmexp.hpp
        include/rpp/cpu/ops/single_thread/ft_inplace_fma.hpp
        include/rpp/cpu/ops/single_thread/ft_inplace_mul.hpp
        include/rpp/cpu/ops/single_thread/ft_log.hpp
        include/rpp/cpu/ops/single_thread/ft_mul.hpp
        include/rpp/cpu/ops/single_thread/st_adj_mul.hpp
        include/rpp/cpu/ops/single_thread/st_fma.hpp
        include/rpp/cpu/ops/single_thread/st_inplace_fma.hpp
        include/rpp/cpu/ops/single_thread/st_mul.hpp
        include/rpp/cpu/ops/single_thread/tensor_add_identity.hpp
        include/rpp/cpu/ops/single_thread/tensor_antipode.hpp
        include/rpp/cpu/ops/single_thread/tensor_reflect.hpp
        include/rpp/cpu/ops/single_thread/tensor_set_identity.hpp
        include/rpp/cpu/ops/single_thread/vector_assign.hpp
        include/rpp/cpu/ops/single_thread/vector_inplace_add.hpp
        include/rpp/cpu/ops/single_thread/vector_set_constant.hpp
        include/rpp/cpu/ops/single_thread/vector_set_zero.hpp
)
add_library(RPP::RPP ALIAS RPP)

target_include_directories(RPP INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(RPPStub STATIC src/stub.cpp)
target_link_libraries(RPPStub PRIVATE RPP)

#
#target_sources(RPP INTERFACE FILE_SET HEADERS BASE_DIRS include FILES
#        include/rpp/config/compiler.h
#        include/rpp/config/target.h
#        include/rpp/dense/batch.hpp
#        include/rpp/dense/views.hpp
#        include/rpp/architecture.hpp
#        include/rpp/basis.hpp
#        include/rpp/config.h
#        include/rpp/integer_maths.hpp
#        include/rpp/operations.hpp
#        include/rpp/utility.hpp
#        include/rpp/cpu/strategies.hpp
#        include/rpp/cpu/ops/single_thread.hpp
#        include/rpp/cpu/ops/single_thread/ft_adj_lmul.hpp
#        include/rpp/cpu/ops/single_thread/ft_adj_rmul.hpp
#        include/rpp/cpu/ops/single_thread/ft_exp.hpp
#        include/rpp/cpu/ops/single_thread/ft_fma.hpp
#        include/rpp/cpu/ops/single_thread/ft_fmexp.hpp
#        include/rpp/cpu/ops/single_thread/ft_inplace_fma.hpp
#        include/rpp/cpu/ops/single_thread/ft_inplace_mul.hpp
#        include/rpp/cpu/ops/single_thread/ft_log.hpp
#        include/rpp/cpu/ops/single_thread/ft_mul.hpp
#        include/rpp/cpu/ops/single_thread/st_adj_mul.hpp
#        include/rpp/cpu/ops/single_thread/st_fma.hpp
#        include/rpp/cpu/ops/single_thread/st_inplace_fma.hpp
#        include/rpp/cpu/ops/single_thread/st_mul.hpp
#        include/rpp/cpu/ops/single_thread/tensor_add_identity.hpp
#        include/rpp/cpu/ops/single_thread/tensor_antipode.hpp
#        include/rpp/cpu/ops/single_thread/tensor_reflect.hpp
#        include/rpp/cpu/ops/single_thread/tensor_set_identity.hpp
#        include/rpp/cpu/ops/single_thread/vector_assign.hpp
#        include/rpp/cpu/ops/single_thread/vector_inplace_add.hpp
#        include/rpp/cpu/ops/single_thread/vector_set_constant.hpp
#        include/rpp/cpu/ops/single_thread/vector_set_zero.hpp
#)


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
        rpp/gpu/ops/block/vector_inplace_add.hpp
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
