cmake_minimum_required(VERSION 4.2)


project(RoughPathPrimitives
        VERSION 0.0.0
        LANGUAGES CXX
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()


option(RPP_ENABLE_CUDA "Enable cuda support" OFF)
option(RPP_ENABLE_TESTS "Enable building tests" ${PROJECT_IS_TOP_LEVEL})
option(RPP_ENABLE_BENCHMARKS "Enable building benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(RPP_ADD_STUB_TARGET "Add a stub target that includes all headers" ${PROJECT_IS_TOP_LEVEL})


include(GNUInstallDirs)


add_library(RPP INTERFACE)
add_library(RPP::RPP ALIAS RPP)

target_include_directories(RPP INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

#add_library(RPPStub STATIC src/stub.cpp)
#target_link_libraries(RPPStub PRIVATE RPP)


set(RPP_BASIC_OPS
        ft_adj_lmul
        ft_adj_rmul
        ft_fma
        ft_inplace_fma
        ft_inplace_mul
        ft_mul
        lie_to_tensor
        sparse_matrix_vector
        st_adj_mul
        st_fma
        st_inplace_fma
        st_mul
        tensor_add_identity
        tensor_antipode
        tensor_generalised_antipode
        tensor_pairing
        tensor_reflect
        tensor_set_identity
        tensor_to_lie
        vector_add
        vector_assign
        vector_inplace_add
        vector_scalar_multiply
        vector_set_constant
)

set(RPP_INTERMEDIATE_OPS
        ft_exp
        ft_fmexp
        ft_log
)


function(rpp_make_op_file_list _out)
    set(options PRIMARY)
    set(one_param_args PLATFORM STRATEGY)
    set(multi_param_args ADDITIONAL)
    cmake_parse_arguments(PARSE_ARGV 1 "ARG" "${options}" "${one_param_args}" "${multi_param_args}")

    set(_working)
    set(_missing)
    if (ARG_PRIMARY)
        set(_prefix "include/rpp/operations")
    else()
        set(_prefix "include/rpp/${ARG_PLATFORM}/operations/${ARG_STRATEGY}")
    endif()
    foreach(op IN LISTS RPP_BASIC_OPS)
        set(_path "${_prefix}/basic/${op}.hpp")
        message(TRACE "Adding operation header file: ${_path}")
        if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${_path}")
            list(APPEND _missing "${_path}")
        endif()
        list(APPEND _working ${_path})
    endforeach()

    foreach(op IN LISTS RPP_INTERMEDIATE_OPS)
        set(_path "${_prefix}/intermediate/${op}.hpp")
        message(TRACE "Adding operation header file: ${_path}")
        if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${_path}")
            list(APPEND _missing "${_path}")
        endif()
        list(APPEND _working ${_path})
    endforeach()

    foreach(addition IN LISTS ARG_ADDITIONAL)
        set(_path "${_prefix}/${addition}")
        message(TRACE "Adding additional operation header: ${_path}")
        if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/${_path}")
            list(APPEND _missing "${_path}")
        endif()
        list(APPEND _working ${_path})
    endforeach()

    if (NOT "${_missing}" STREQUAL "")
        list(JOIN _missing "\n" _missing_paths)
        message(WARNING
                "The following operation files could not be found:\n${_missing_paths}")
    endif()

    set(${_out} ${_working} PARENT_SCOPE)
endfunction()


rpp_make_op_file_list(RPP_OPS PRIMARY ADDITIONAL base_operation.hpp)
rpp_make_op_file_list(RPP_CPU_ST_OPS PLATFORM cpu STRATEGY single_thread
    ADDITIONAL
        basic/detail/ft_multiply.hpp
        detail/batch_wrapper.hpp
)

target_sources(RPP INTERFACE FILE_SET HEADERS BASE_DIRS include FILES
        include/rpp/basis/basis_pack.hpp
        include/rpp/basis/basis_tags.hpp
        include/rpp/basis/graded_basis.hpp
        include/rpp/basis/hall_basis.hpp
        include/rpp/basis/lie_basis.hpp
        include/rpp/basis/lie_multiplication.hpp
        include/rpp/basis/tensor_basis.hpp
        include/rpp/config/compiler.h
        include/rpp/config/target.h
        include/rpp/support/error.hpp
        include/rpp/support/span.hpp
        include/rpp/support/strided_iterator.hpp
        include/rpp/dense/batch.hpp
        include/rpp/dense/views.hpp
        include/rpp/sparse/detail/compressed_matrix.hpp
        include/rpp/sparse/matrix.hpp
        include/rpp/sparse/l2t_matrix.hpp
        include/rpp/sparse/t2l_matrix.hpp
        include/rpp/architecture.hpp
        include/rpp/basis.hpp
        include/rpp/config.h
        include/rpp/operations.hpp
        include/rpp/utility.hpp
        include/rpp/cpu/strategies.hpp
        ${RPP_OPS}
        ${RPP_CPU_ST_OPS}
)





if (RPP_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    rpp_make_op_file_list(RPP_GPU_BLOCK_OPS PLATFORM gpu STRATEGY block
            ADDITIONAL
            strategy.hpp
            basic/detail/ft_adjoint_multiply.hpp
            basic/detail/ft_multiply.hpp
            basic/detail/st_multiply.hpp
    )

    set(RPP_GPU_SOURCES
            include/rpp/gpu/architecture.hpp
            include/rpp/gpu/strategies.hpp
            ${RPP_GPU_BLOCK_OPS}
    )

    target_sources(RPP INTERFACE FILE_SET HEADERS BASE_DIRS include FILES
        ${RPP_GPU_SOURCES}
    )
    set_source_files_properties(${RPP_GPU_SOURCES} PROPERTIES LANGUAGE CUDA)
endif()

if (RPP_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if (RPP_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()


if(RPP_ADD_STUB_TARGET)
    add_library(RPP_Stub STATIC)
    target_link_libraries(RPP_Stub PRIVATE RPP)

    get_target_property(_headers RPP HEADER_SET)
    set(_include)

    foreach(_hdr IN LISTS _headers)
        cmake_path(RELATIVE_PATH _hdr BASE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include")
        list(APPEND _include "#include <${_hdr}>")
    endforeach()
    list(JOIN _include "\n" _include_str)

    if(RPP_ENABLE_CUDA)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/stub.cu" "${_include_str}")
        target_sources(RPP_Stub PRIVATE
                ${CMAKE_CURRENT_BINARY_DIR}/stub.cu)
    else()
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/stub.cpp" "${_include_str}")
        target_sources(RPP_Stub PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/stub.cpp)
    endif()
endif()