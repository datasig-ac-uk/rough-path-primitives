#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/sparse_matrix_vector.hpp>
#include <rpp/gpu/block/operations/linalg/sparse_matrix_vector.hpp>
#include <rpp/sparse/detail/compressed_matrix.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

struct SparseMatrixData {
    using Helper = rpp::tests::GpuBlockTestHelper;

    Helper::HostVector<Helper::Scalar> values;
    Helper::HostVector<Helper::Index> indices;
    Helper::HostVector<Helper::Index> offsets;
};

SparseMatrixData make_csr_data() {
    return {
        {2.0, -1.0, 0.5, 3.0, 1.5, -2.0}, {0, 2, 1, 0, 3, 2}, {0, 2, 3, 5, 6}};
}

SparseMatrixData make_csc_data() {
    return {
        {2.0, 3.0, 0.5, -1.0, -2.0, 1.5}, {0, 2, 1, 0, 3, 2}, {0, 2, 3, 5, 6}};
}

TEST(GpuBlockSparseMatrixVectorProductTests,
     CsrMatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    auto const basis_data = Helper::BasisData(1, 3);
    auto const& basis = basis_data.basis;
    auto const cpu_strategy = Helper::cpu_strategy();
    auto const gpu_strategy = Helper::gpu_strategy();
    auto constexpr alpha = Helper::Scalar{-0.75};
    auto matrix_data = make_csr_data();

    auto expected = Helper::make_batch(1, basis);
    auto actual = expected;
    auto const arg = Helper::make_batch(2, basis, Helper::Scalar{0.01});

    auto const host_matrix =
        rpp::sparse::make_csr_matrix(Helper::host_data(matrix_data.values),
                                     Helper::host_data(matrix_data.indices),
                                     Helper::host_data(matrix_data.offsets),
                                     matrix_data.values.size(),
                                     basis.size(),
                                     basis.size());

    Helper::DeviceVector<Helper::Scalar> device_actual(actual);
    Helper::DeviceVector<Helper::Scalar> device_arg(arg);
    rpp::gpu::DeviceLaunchConfig launch_config;
    launch_config.stream = nullptr;
    auto const err = rpp::ops::sparse_matrix_vector_product(
        gpu_strategy,
        std::move(launch_config),
        Helper::device_vector_batch(device_actual, basis),
        Helper::device_vector_batch(device_arg, basis),
        basis,
        basis,
        Helper::tensor_count,
        host_matrix,
        alpha);
    ASSERT_TRUE(static_cast<bool>(err)) << err.message();
    RPP_CUDA_ASSERT(cudaDeviceSynchronize());

    auto const cpu_err = rpp::ops::sparse_matrix_vector_product(
        cpu_strategy,
        Helper::CpuStrategy::LaunchConfig{},
        Helper::host_vector_batch(expected, basis),
        Helper::host_vector_batch(arg, basis),
        basis,
        basis,
        Helper::tensor_count,
        host_matrix,
        alpha);
    ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

    actual = Helper::copy_to_host(device_actual);
    Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
}

TEST(GpuBlockSparseMatrixVectorProductTests,
     CscMatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    auto const basis_data = Helper::BasisData(1, 3);
    auto const& basis = basis_data.basis;
    auto const cpu_strategy = Helper::cpu_strategy();
    auto const gpu_strategy = Helper::gpu_strategy();
    auto constexpr alpha = Helper::Scalar{1.25};
    auto matrix_data = make_csc_data();

    auto expected = Helper::make_batch(1, basis);
    auto actual = expected;
    auto const arg = Helper::make_batch(2, basis, Helper::Scalar{0.01});

    auto const host_matrix =
        rpp::sparse::make_csc_matrix(Helper::host_data(matrix_data.values),
                                     Helper::host_data(matrix_data.indices),
                                     Helper::host_data(matrix_data.offsets),
                                     matrix_data.values.size(),
                                     basis.size(),
                                     basis.size());

    Helper::DeviceVector<Helper::Scalar> device_actual(actual);
    Helper::DeviceVector<Helper::Scalar> device_arg(arg);
    rpp::gpu::DeviceLaunchConfig launch_config;
    launch_config.stream = nullptr;
    auto const err = rpp::ops::sparse_matrix_vector_product(
        gpu_strategy,
        std::move(launch_config),
        Helper::device_vector_batch(device_actual, basis),
        Helper::device_vector_batch(device_arg, basis),
        basis,
        basis,
        Helper::tensor_count,
        host_matrix,
        alpha);
    ASSERT_TRUE(static_cast<bool>(err)) << err.message();
    RPP_CUDA_ASSERT(cudaDeviceSynchronize());

    auto const cpu_err = rpp::ops::sparse_matrix_vector_product(
        cpu_strategy,
        Helper::CpuStrategy::LaunchConfig{},
        Helper::host_vector_batch(expected, basis),
        Helper::host_vector_batch(arg, basis),
        basis,
        basis,
        Helper::tensor_count,
        host_matrix,
        alpha);
    ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

    actual = Helper::copy_to_host(device_actual);
    Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
}

} // namespace
