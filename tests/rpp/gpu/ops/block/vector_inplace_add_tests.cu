#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/vector_inplace_add.hpp>
#include <rpp/gpu/operations/block/basic/vector_inplace_add.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockVectorInplaceAddTests, MatchesCpuForSingleElementBatches)
{
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();
        auto constexpr alpha = Helper::Scalar{-1.75};

        auto expected = Helper::make_batch(1, basis);
        auto actual = expected;
        auto const rhs = Helper::make_batch(2, basis);

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_rhs(rhs);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::block::vector_inplace_add_kernel<<<
            Helper::tensor_count,
            gpu_strategy.block_size,
            0
        >>>(
            Helper::device_vector_batch(device_actual, basis),
            Helper::device_vector_batch(device_rhs, basis),
            device_basis.basis,
            gpu_strategy,
            Helper::tensor_count,
            alpha
        );
        RPP_CUDA_ASSERT(cudaGetLastError());
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        rpp::cpu::single_thread::vector_inplace_add_kernel(
            Helper::host_vector_batch(expected, basis),
            Helper::host_vector_batch(rhs, basis),
            basis,
            cpu_strategy,
            Helper::tensor_count,
            alpha
        );

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-5});
    }
}

} // namespace
