include(FetchContent)

FetchContent_Declare(
        LibalgebraLite
        GIT_REPOSITORY https://github.com/datasig-ac-uk/libalgebra-lite.git
        GIT_TAG 5adf56d2211df9faec7b0990ce332cfae0c42396
)

set(LIBALGEBRA_LITE_RATIONAL_COEFFS ON CACHE INTERNAL "Use rational coefficients in CPU algebra tests")
FetchContent_MakeAvailable(LibalgebraLite)

rpp_add_test(rpp_cpu_single_thread_algebra_tests
        cpu_kernel_wrapper_test_helper.hpp
        polynomial_tensor_helper.hpp
        ft_adj_lmul_tests.cpp
        ft_adj_rmul_tests.cpp
        ft_exp_log_tests.cpp
        ft_fma_tests.cpp
        ft_fmexp_tests.cpp
        ft_inplace_fma_tests.cpp
        ft_inplace_mul_tests.cpp
        ft_mul_tests.cpp
        st_adj_mul_tests.cpp
        st_fma_tests.cpp
        st_inplace_fma_tests.cpp
        st_mul_tests.cpp
        tensor_antipode_tests.cpp
        tensor_identity_tests.cpp
        tensor_reflect_tests.cpp
        vector_assign_tests.cpp
        vector_inplace_add_tests.cpp
        vector_set_tests.cpp
)
target_link_libraries(rpp_cpu_single_thread_algebra_tests
        PRIVATE
            Libalgebra_lite::Libalgebra_lite
)
