#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_inplace_mul.hpp>
#include <rpp/gpu/block/operations/basic/ft_inplace_mul.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

template <typename BatchLhs,
          typename BatchRhs,
          typename Basis,
          typename Strategy>
__global__ void
ft_inplace_mul_truncated_rhs_kernel(BatchLhs batch_lhs,
                                    BatchRhs batch_rhs,
                                    Basis basis,
                                    Strategy strategy,
                                    typename Strategy::Index n_tensors,
                                    typename Strategy::Accum beta) {
    extern __shared__ std::byte smem_bytes[];

    auto const ctx = strategy.make_context(smem_bytes);
    auto const my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) {
        return;
    }

    rpp::ops::FTInplaceMul<Strategy> op;
    auto lhs = batch_lhs.view(basis, my_index);
    auto rhs = batch_rhs.view(basis, my_index).truncate(1, basis.depth);
    op(ctx, lhs, rhs, beta);
}

TEST(GpuBlockFtInplaceMulTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    using GpuOp = rpp::ops::FTInplaceMul<Helper::GpuStrategy>;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();
        auto constexpr beta = Helper::Scalar{0.75};

        auto expected = Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto actual = expected;
        auto const rhs = Helper::make_batch(2, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_rhs(rhs);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_inplace_mul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_rhs, basis),
            basis,
            Helper::tensor_count,
            beta);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
            Helper::launch_cpu([&](auto const& strategy, auto config) {
                return rpp::ops::ft_inplace_mul(
                    strategy,
                    std::move(config),
                    Helper::host_tensor_batch(expected, basis),
                    Helper::host_tensor_batch(rhs, basis),
                    basis,
                    Helper::tensor_count,
                    beta);
            });
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

TEST(GpuBlockFtInplaceMulTests, ZerosUnitCoefficientWhenRhsHasNoUnitTerm) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    using GpuOp = rpp::ops::FTInplaceMul<Helper::GpuStrategy>;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();
        auto constexpr beta = Helper::Scalar{0.75};

        auto expected = Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto actual = expected;
        auto const rhs = Helper::make_batch(2, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_rhs(rhs);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;

        auto const rhs_batch = Helper::device_tensor_batch(device_rhs, basis);
        auto const rhs_truncated = rpp::make_tensor_batch(
            rhs_batch.data(), rhs_batch.layout(), basis, 1, basis.depth);

        auto const err = rpp::ops::ft_inplace_mul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            rhs_truncated,
            basis,
            Helper::tensor_count,
            beta);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_ctx = Helper::CpuStrategy::make_context(nullptr);
        auto expected_view = Helper::host_tensor_batch(expected, basis).view(0);
        auto rhs_view = Helper::host_tensor_batch(rhs, basis).view(0);
        auto rhs_trunc = rhs_view.truncate(1, basis.depth);
        rpp::ops::FTInplaceMul<Helper::CpuStrategy>{}(
            cpu_ctx, expected_view, rhs_trunc, beta);

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(
            actual[0], Helper::Scalar{0}, Helper::Scalar{1.5e-5});
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
