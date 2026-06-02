#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/st_adj_mul.hpp>
#include <rpp/gpu/block/operations/basic/st_adj_mul.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockStAdjMulTests, MatchesCpuForSingleElementBatches) {
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
        auto const err = rpp::ops::st_adj_mul(
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
             rpp::ops::st_adj_mul(cpu_strategy,
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

TEST(GpuBlockStAdjMulTests, IdentityOperatorMatchesCpuForTruncatedView) {
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
        auto const err = rpp::ops::st_adj_mul(
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

        auto const cpu_err = rpp::ops::st_adj_mul(
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
