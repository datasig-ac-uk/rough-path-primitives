#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/intermediate/ft_log.hpp>
#include <rpp/gpu/block/operations/intermediate/ft_log.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockFtLogTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    using GpuOp = rpp::ops::FTLog<Helper::GpuStrategy>;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_zero_batch(basis);
        auto actual = expected;
        auto const arg = Helper::make_batch(1, basis, Helper::Scalar{0.005});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err =
            rpp::ops::ft_log(gpu_strategy,
                             std::move(launch_config),
                             Helper::device_tensor_batch(device_actual, basis),
                             Helper::device_tensor_batch(device_arg, basis),
                             basis,
                             Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err = Helper::launch_cpu([&](auto const& strategy,
                                                    auto config) {
            return rpp::ops::ft_log(strategy,
                                    std::move(config),
                                    Helper::host_tensor_batch(expected, basis),
                                    Helper::host_tensor_batch(arg, basis),
                                    basis,
                                    Helper::tensor_count);
        });
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(
            actual[0], Helper::Scalar{0}, Helper::Scalar{1.5e-5});
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
