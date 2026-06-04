#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_adj_rmul.hpp>
#include <rpp/gpu/block/operations/basic/ft_adj_rmul.hpp>
#include <rpp/gpu/block/operations/basic/ft_mul.hpp>
#include <rpp/gpu/block/operations/basic/tensor_pairing.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

void expect_right_adjoint_pairing_identity(
    rpp::tests::GpuBlockTestHelper::Basis const& basis,
    rpp::tests::GpuBlockTestHelper::GpuStrategy const& gpu_strategy,
    rpp::tests::GpuBlockTestHelper::HostVector<
        rpp::tests::GpuBlockTestHelper::Scalar> const& op,
    rpp::tests::GpuBlockTestHelper::HostVector<
        rpp::tests::GpuBlockTestHelper::Scalar> const& t,
    rpp::tests::GpuBlockTestHelper::HostVector<
        rpp::tests::GpuBlockTestHelper::Scalar> const& arg,
    rpp::tests::GpuBlockTestHelper::Scalar tolerance) {
    using Helper = rpp::tests::GpuBlockTestHelper;

    auto adjoint = Helper::make_zero_batch(basis);
    auto product = Helper::make_zero_batch(basis);

    Helper::DeviceVector<Helper::Scalar> device_adjoint(adjoint);
    Helper::DeviceVector<Helper::Scalar> device_product(product);
    Helper::DeviceVector<Helper::Scalar> device_op(op);
    Helper::DeviceVector<Helper::Scalar> device_t(t);
    Helper::DeviceVector<Helper::Scalar> device_arg(arg);
    Helper::DeviceVector<Helper::Scalar> device_lhs_pairing(1);
    Helper::DeviceVector<Helper::Scalar> device_rhs_pairing(1);

    rpp::gpu::DeviceLaunchConfig launch_config;
    launch_config.stream = nullptr;
    auto const adj_err = rpp::ops::ft_adj_rmul(
        gpu_strategy,
        launch_config,
        Helper::device_tensor_batch(device_adjoint, basis),
        Helper::device_tensor_batch(device_op, basis),
        Helper::device_tensor_batch(device_arg, basis),
        basis,
        Helper::tensor_count);
    ASSERT_TRUE(static_cast<bool>(adj_err)) << adj_err.message();

    auto const mul_err = rpp::ops::ft_mul(
        gpu_strategy,
        launch_config,
        Helper::device_tensor_batch(device_product, basis),
        Helper::device_tensor_batch(device_t, basis),
        Helper::device_tensor_batch(device_op, basis),
        basis,
        Helper::tensor_count);
    ASSERT_TRUE(static_cast<bool>(mul_err)) << mul_err.message();

    auto const lhs_err = rpp::ops::tensor_pairing(
        gpu_strategy,
        launch_config,
        Helper::device_scalar_batch(device_lhs_pairing),
        Helper::device_tensor_batch(device_adjoint, basis),
        Helper::device_tensor_batch(device_t, basis),
        basis,
        Helper::tensor_count);
    ASSERT_TRUE(static_cast<bool>(lhs_err)) << lhs_err.message();

    auto const rhs_err = rpp::ops::tensor_pairing(
        gpu_strategy,
        launch_config,
        Helper::device_scalar_batch(device_rhs_pairing),
        Helper::device_tensor_batch(device_arg, basis),
        Helper::device_tensor_batch(device_product, basis),
        basis,
        Helper::tensor_count);
    ASSERT_TRUE(static_cast<bool>(rhs_err)) << rhs_err.message();

    RPP_CUDA_ASSERT(cudaDeviceSynchronize());

    auto const lhs_pairing = Helper::copy_to_host(device_lhs_pairing);
    auto const rhs_pairing = Helper::copy_to_host(device_rhs_pairing);
    ASSERT_EQ(lhs_pairing.size(), std::size_t{1});
    ASSERT_EQ(rhs_pairing.size(), std::size_t{1});
    Helper::expect_near(lhs_pairing[0], rhs_pairing[0], tolerance);
}

TEST(GpuBlockFtAdjRMulTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_zero_batch(basis);
        auto actual = expected;
        auto const op = Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto const arg = Helper::make_batch(2, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_op(op);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_adj_rmul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_op, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
             rpp::ops::ft_adj_rmul(cpu_strategy,
                                    Helper::CpuStrategy::LaunchConfig{},
                    Helper::host_tensor_batch(expected, basis),
                    Helper::host_tensor_batch(op, basis),
                    Helper::host_tensor_batch(arg, basis),
                    basis,
                    Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

TEST(GpuBlockFtAdjRMulTests, SatisfiesAdjointPairingCriterionOnGpu) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    constexpr unsigned seeds[][3] = {
        {1, 2, 3},
        {5, 8, 13},
        {21, 34, 55},
        {89, 144, 233},
    };

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = Helper::gpu_strategy();

        for (auto const& triple : seeds) {
            auto const op = Helper::make_batch(triple[0], basis, Helper::Scalar{0.01});
            auto const t = Helper::make_batch(triple[1], basis, Helper::Scalar{0.01});
            auto const arg = Helper::make_batch(triple[2], basis, Helper::Scalar{0.01});

            expect_right_adjoint_pairing_identity(
                basis, gpu_strategy, op, t, arg, Helper::Scalar{2.5e-4});
        }
    }
}

TEST(GpuBlockFtAdjRMulTests, IdentityOperatorMatchesCpuForTruncatedView) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_zero_batch(basis);
        auto actual = expected;
        auto op = Helper::make_zero_batch(basis);
        auto const arg = Helper::make_batch(7, basis, Helper::Scalar{0.01});
        op[0] = Helper::Scalar{1};

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_op(op);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_adj_rmul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            rpp::make_tensor_batch(
                Helper::device_data(device_op),
                basis.size(),
                Helper::Degree{0},
                Helper::Degree{0}),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err = rpp::ops::ft_adj_rmul(
            cpu_strategy,
            Helper::CpuStrategy::LaunchConfig{},
            Helper::host_tensor_batch(expected, basis),
            rpp::make_tensor_batch(
                Helper::host_data(op),
                basis.size(),
                basis,
                Helper::Degree{0},
                Helper::Degree{0}),
            Helper::host_tensor_batch(arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
