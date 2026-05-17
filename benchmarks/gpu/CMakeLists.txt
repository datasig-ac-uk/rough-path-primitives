function(rpp_add_gpu_benchmark target)
    add_executable(${target}
            ${ARGN}
    )

    target_link_libraries(${target}
            PRIVATE
                RPP::GPU
                CUDA::cudart
                benchmark::benchmark_main
    )

    target_include_directories(${target}
            PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}
    )

    set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )

    target_compile_options(${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

endfunction()

rpp_add_gpu_benchmark(gpu_sparse_matrix_vector_product_bench
        sparse_matrix_vector_product_bench.cu
)
