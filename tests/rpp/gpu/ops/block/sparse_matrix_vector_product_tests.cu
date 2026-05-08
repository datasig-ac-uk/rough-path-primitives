#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/sparse_matrix_vector.hpp>
#include <rpp/gpu/operations/block/basic/sparse_matrix_vector.hpp>
#include <rpp/sparse/detail/compressed_matrix.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

struct SparseMatrixData {
    using Helper = rpp::tests::GpuBlockTestHelper;

    Helper::HostVector<Helper::Scalar> values;
    Helper::HostVector<Helper::Index> indices;
    Helper::HostVector<Helper::Index> offsets;
};

SparseMatrixData make_csr_data()
{
    return {
        {2.0, -1.0, 0.5, 3.0, 1.5, -2.0},
        {0, 2, 1, 0, 3, 2},
        {0, 2, 3, 5, 6}
    };
}

SparseMatrixData make_csc_data()
{
    return {
        {2.0, 3.0, 0.5, -1.0, -2.0, 1.5},
        {0, 2, 1, 0, 3, 2},
        {0, 2, 3, 5, 6}
    };
}

TEST(GpuBlockSparseMatrixVectorProductTests, CsrMatchesCpuForSingleElementBatches)
{
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

    auto const host_matrix = rpp::sparse::make_csr_matrix(
        matrix_data.values.data(),
        matrix_data.indices.data(),
        matrix_data.offsets.data(),
        matrix_data.values.size(),
        basis.size(),
        basis.size()
    );

    Helper::DeviceVector<Helper::Scalar> device_actual(actual);
    Helper::DeviceVector<Helper::Scalar> device_arg(arg);
    Helper::DeviceVector<Helper::Scalar> device_values(matrix_data.values);
    Helper::DeviceVector<Helper::Index> device_indices(matrix_data.indices);
    Helper::DeviceVector<Helper::Index> device_offsets(matrix_data.offsets);
    Helper::DeviceBasis device_basis(basis_data);

    auto const device_matrix = rpp::sparse::make_csr_matrix(
        Helper::device_data(device_values),
        Helper::device_data(device_indices),
        Helper::device_data(device_offsets),
        matrix_data.values.size(),
        basis.size(),
        basis.size()
    );

    rpp::gpu::block::sparse_matrix_vector_product_kernel<<<
        Helper::tensor_count,
        gpu_strategy.block_size,
        0
    >>>(
        Helper::device_vector_batch(device_actual, basis),
        device_matrix,
        Helper::device_vector_batch(device_arg, basis),
        device_basis.basis,
        device_basis.basis,
        gpu_strategy,
        Helper::tensor_count,
        alpha
    );
    RPP_CUDA_ASSERT(cudaGetLastError());
    RPP_CUDA_ASSERT(cudaDeviceSynchronize());

    rpp::cpu::single_thread::sparse_matrix_vector_product_kernel(
        Helper::host_vector_batch(expected, basis),
        host_matrix,
        Helper::host_vector_batch(arg, basis),
        basis,
        basis,
        cpu_strategy,
        Helper::tensor_count,
        alpha
    );

    actual = Helper::copy_to_host(device_actual);
    Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
}

TEST(GpuBlockSparseMatrixVectorProductTests, CscMatchesCpuForSingleElementBatches)
{
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

    auto const host_matrix = rpp::sparse::make_csc_matrix(
        matrix_data.values.data(),
        matrix_data.indices.data(),
        matrix_data.offsets.data(),
        matrix_data.values.size(),
        basis.size(),
        basis.size()
    );

    Helper::DeviceVector<Helper::Scalar> device_actual(actual);
    Helper::DeviceVector<Helper::Scalar> device_arg(arg);
    Helper::DeviceVector<Helper::Scalar> device_values(matrix_data.values);
    Helper::DeviceVector<Helper::Index> device_indices(matrix_data.indices);
    Helper::DeviceVector<Helper::Index> device_offsets(matrix_data.offsets);
    Helper::DeviceBasis device_basis(basis_data);

    auto const device_matrix = rpp::sparse::make_csc_matrix(
        Helper::device_data(device_values),
        Helper::device_data(device_indices),
        Helper::device_data(device_offsets),
        matrix_data.values.size(),
        basis.size(),
        basis.size()
    );

    using Op = rpp::ops::SparseMatrixVectorProduct<rpp::gpu::strategies::BlockStrategy<Helper::Scalar>, rpp::sparse::CSCMatrix>;
    auto smem_bytes = Op::scratch_space_size(gpu_strategy, basis);

    rpp::gpu::block::sparse_matrix_vector_product_kernel<<<
        Helper::tensor_count,
        gpu_strategy.block_size,
        smem_bytes
    >>>(
        Helper::device_vector_batch(device_actual, basis),
        device_matrix,
        Helper::device_vector_batch(device_arg, basis),
        device_basis.basis,
        device_basis.basis,
        gpu_strategy,
        Helper::tensor_count,
        alpha
    );
    RPP_CUDA_ASSERT(cudaGetLastError());
    RPP_CUDA_ASSERT(cudaDeviceSynchronize());

    rpp::cpu::single_thread::sparse_matrix_vector_product_kernel(
        Helper::host_vector_batch(expected, basis),
        host_matrix,
        Helper::host_vector_batch(arg, basis),
        basis,
        basis,
        cpu_strategy,
        Helper::tensor_count,
        alpha
    );

    actual = Helper::copy_to_host(device_actual);
    Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
}

} // namespace
