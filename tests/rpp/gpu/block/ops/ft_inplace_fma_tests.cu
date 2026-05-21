#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_fma.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_inplace_fma.hpp>
#include <rpp/gpu/block/operations/basic/ft_inplace_fma.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockFtInplaceFmaTests,
     AEqualsBCPlusAMatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    using GpuOp =
        rpp::ops::FTInplaceFma<Helper::GpuStrategy,
                               rpp::ops::FTInplaceFMAType::AEqualsBCPlusA>;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();
        auto constexpr alpha = Helper::Scalar{0.5};
        auto constexpr beta = Helper::Scalar{-1.25};

        auto expected = Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto actual = expected;
        auto const b = Helper::make_batch(2, basis, Helper::Scalar{0.01});
        auto const c = Helper::make_batch(3, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_b(b);
        Helper::DeviceVector<Helper::Scalar> device_c(c);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_inplace_fma<
            rpp::ops::FTInplaceFMAType::AEqualsBCPlusA>(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_b, basis),
            Helper::device_tensor_batch(device_c, basis),
            basis,
            Helper::tensor_count,
            alpha,
            beta);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
            Helper::launch_cpu([&](auto const& strategy, auto config) {
                return rpp::ops::ft_inplace_fma<
                    rpp::ops::FTInplaceFMAType::AEqualsBCPlusA>(
                    strategy,
                    std::move(config),
                    Helper::host_tensor_batch(expected, basis),
                    Helper::host_tensor_batch(b, basis),
                    Helper::host_tensor_batch(c, basis),
                    basis,
                    Helper::tensor_count,
                    alpha,
                    beta);
            });
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

TEST(GpuBlockFtInplaceFmaTests, OrderedVariantsMatchCpuFmaReference) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    using GpuABOp =
        rpp::ops::FTInplaceFma<Helper::GpuStrategy,
                               rpp::ops::FTInplaceFMAType::AEqualsABPlusC>;
    using GpuBAOp =
        rpp::ops::FTInplaceFma<Helper::GpuStrategy,
                               rpp::ops::FTInplaceFMAType::AEqualsBAPlusC>;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();
        auto constexpr alpha = Helper::Scalar{-0.5};
        auto constexpr beta = Helper::Scalar{0.75};

        auto const initial_a =
            Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto const b = Helper::make_batch(2, basis, Helper::Scalar{0.01});
        auto const c = Helper::make_batch(3, basis, Helper::Scalar{0.01});

        auto expected_ab = Helper::make_zero_batch(basis);
        auto actual_ab = initial_a;
        Helper::DeviceVector<Helper::Scalar> device_ab(actual_ab);
        Helper::DeviceVector<Helper::Scalar> device_b(b);
        Helper::DeviceVector<Helper::Scalar> device_c(c);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config_ab;
        launch_config_ab.stream = nullptr;
        auto const err_ab = rpp::ops::ft_inplace_fma<
            rpp::ops::FTInplaceFMAType::AEqualsABPlusC>(
            gpu_strategy,
            std::move(launch_config_ab),
            Helper::device_tensor_batch(device_ab, basis),
            Helper::device_tensor_batch(device_b, basis),
            Helper::device_tensor_batch(device_c, basis),
            basis,
            Helper::tensor_count,
            alpha,
            beta);
        ASSERT_TRUE(static_cast<bool>(err_ab)) << err_ab.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err_ab =
            Helper::launch_cpu([&](auto const& strategy, auto config) {
                return rpp::ops::ft_fma(
                    strategy,
                    std::move(config),
                    Helper::host_tensor_batch(expected_ab, basis),
                    Helper::host_tensor_batch(c, basis),
                    Helper::host_tensor_batch(initial_a, basis),
                    Helper::host_tensor_batch(b, basis),
                    basis,
                    Helper::tensor_count,
                    alpha,
                    beta);
            });
        ASSERT_TRUE(static_cast<bool>(cpu_err_ab)) << cpu_err_ab.message();

        actual_ab = Helper::copy_to_host(device_ab);
        Helper::expect_near(actual_ab, expected_ab, Helper::Scalar{1.5e-4});

        auto expected_ba = Helper::make_zero_batch(basis);
        auto actual_ba = initial_a;
        Helper::DeviceVector<Helper::Scalar> device_ba(actual_ba);

        rpp::gpu::DeviceLaunchConfig launch_config_ba;
        launch_config_ba.stream = nullptr;
        auto const err_ba = rpp::ops::ft_inplace_fma<
            rpp::ops::FTInplaceFMAType::AEqualsBAPlusC>(
            gpu_strategy,
            std::move(launch_config_ba),
            Helper::device_tensor_batch(device_ba, basis),
            Helper::device_tensor_batch(device_b, basis),
            Helper::device_tensor_batch(device_c, basis),
            basis,
            Helper::tensor_count,
            alpha,
            beta);
        ASSERT_TRUE(static_cast<bool>(err_ba)) << err_ba.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err_ba =
            Helper::launch_cpu([&](auto const& strategy, auto config) {
                return rpp::ops::ft_fma(
                    strategy,
                    std::move(config),
                    Helper::host_tensor_batch(expected_ba, basis),
                    Helper::host_tensor_batch(c, basis),
                    Helper::host_tensor_batch(b, basis),
                    Helper::host_tensor_batch(initial_a, basis),
                    basis,
                    Helper::tensor_count,
                    alpha,
                    beta);
            });
        ASSERT_TRUE(static_cast<bool>(cpu_err_ba)) << cpu_err_ba.message();

        actual_ba = Helper::copy_to_host(device_ba);
        Helper::expect_near(actual_ba, expected_ba, Helper::Scalar{1.5e-4});
    }
}

} // namespace
