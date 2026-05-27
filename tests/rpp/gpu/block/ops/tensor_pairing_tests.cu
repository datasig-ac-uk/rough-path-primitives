#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/tensor_pairing.hpp>
#include <rpp/gpu/block/operations/basic/tensor_pairing.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockTensorPairingTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto const functional_data =
            Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto const arg_data =
            Helper::make_batch(2, basis, Helper::Scalar{0.01});
        Helper::Scalar expected = 0;

        auto expected_batch = Helper::HostVector<Helper::Scalar>(1);
        auto const cpu_err = rpp::ops::tensor_pairing(
            cpu_strategy,
            Helper::CpuStrategy::LaunchConfig{},
            rpp::make_scalar_batch(Helper::host_data(expected_batch)),
            Helper::host_tensor_batch(functional_data, basis),
            Helper::host_tensor_batch(arg_data, basis),
            basis,
            1);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();
        ASSERT_EQ(expected_batch.size(), std::size_t{1});
        expected = expected_batch[0];

        Helper::DeviceVector<Helper::Scalar> device_functional(functional_data);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg_data);
        Helper::DeviceVector<Helper::Scalar> device_actual(1);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_pairing(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_scalar_batch(device_actual),
            Helper::device_tensor_batch(device_functional, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            1);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const actual = Helper::copy_to_host(device_actual);
        ASSERT_EQ(actual.size(), std::size_t{1});
        Helper::expect_near(actual[0], expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
