#set(RPP_GPU_BLOCK_OP_TEST_SOURCES
#        ft_adj_lmul_tests.cu
#        ft_adj_rmul_tests.cu
#        ft_exp_tests.cu
#        ft_fma_tests.cu
#        ft_fmexp_tests.cu
#        ft_inplace_fma_tests.cu
#        ft_inplace_mul_tests.cu
#        ft_log_tests.cu
#        ft_mul_tests.cu
#        gpu_block_test_helper.cuh
#        sparse_matrix_vector_product_tests.cu
#        st_adj_mul_tests.cu
#        st_fma_tests.cu
#        st_inplace_fma_tests.cu
#        st_mul_tests.cu
#        tensor_add_identity_tests.cu
#        tensor_antipode_tests.cu
#        tensor_pairing_tests.cu
#        tensor_reflect_tests.cu
#        tensor_set_identity_tests.cu
#        vector_assign_tests.cu
#        vector_inplace_add_tests.cu
#        vector_set_constant_tests.cu
#        vector_set_zero_tests.cu
#)
#
#rpp_add_test(rpp_gpu_block_ops_tests
#        ${RPP_GPU_BLOCK_OP_TEST_SOURCES}
#)
#target_link_libraries(rpp_gpu_block_ops_tests
#        PRIVATE
#            CUDA::cudart
#)
#set_target_properties(rpp_gpu_block_ops_tests PROPERTIES
#        CUDA_STANDARD 17
#        CUDA_STANDARD_REQUIRED ON
#)
#target_compile_options(rpp_gpu_block_ops_tests
#        PRIVATE
#            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
#)
