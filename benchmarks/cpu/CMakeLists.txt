function(rpp_add_cpu_benchmark target)
    add_executable(${target}
            ${target}.cpp
    )

    target_link_libraries(${target}
            PRIVATE
                RPP::RPP
                benchmark::benchmark_main
    )

    target_include_directories(${target}
            PRIVATE
                ${CMAKE_CURRENT_LIST_DIR}
    )
endfunction()

rpp_add_cpu_benchmark(vector_set_zero_bench)
rpp_add_cpu_benchmark(vector_set_constant_bench)
rpp_add_cpu_benchmark(vector_assign_bench)
rpp_add_cpu_benchmark(vector_inplace_add_bench)
rpp_add_cpu_benchmark(sparse_matrix_vector_product_bench)
rpp_add_cpu_benchmark(tensor_add_identity_bench)
rpp_add_cpu_benchmark(tensor_set_identity_bench)
rpp_add_cpu_benchmark(tensor_pairing_bench)
rpp_add_cpu_benchmark(tensor_antipode_bench)
rpp_add_cpu_benchmark(tensor_reflect_bench)
rpp_add_cpu_benchmark(ft_mul_bench)
rpp_add_cpu_benchmark(ft_inplace_mul_bench)
rpp_add_cpu_benchmark(ft_fma_bench)
rpp_add_cpu_benchmark(ft_inplace_fma_bench)
rpp_add_cpu_benchmark(ft_adj_lmul_bench)
rpp_add_cpu_benchmark(ft_adj_rmul_bench)
rpp_add_cpu_benchmark(st_mul_bench)
rpp_add_cpu_benchmark(st_inplace_fma_bench)
rpp_add_cpu_benchmark(st_fma_bench)
rpp_add_cpu_benchmark(st_adj_mul_bench)
rpp_add_cpu_benchmark(ft_exp_bench)
rpp_add_cpu_benchmark(ft_fmexp_bench)
rpp_add_cpu_benchmark(ft_log_bench)
