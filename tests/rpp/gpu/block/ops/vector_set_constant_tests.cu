#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_set_constant.hpp>
#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockVectorSetConstantTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();
        auto constexpr value = Helper::Scalar{3.25};

        auto expected = Helper::make_batch(1, basis);
        auto actual = expected;

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::vector_set_constant(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_vector_batch(device_actual, basis),
            basis,
            Helper::tensor_count,
            value);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
            Helper::launch_cpu([&](auto const& strategy, auto config) {
                return rpp::ops::vector_set_constant(
                    strategy,
                    std::move(config),
                    Helper::host_vector_batch(expected, basis),
                    basis,
                    Helper::tensor_count,
                    value);
            });
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-5});
    }
}

TEST(GpuBlockVectorSetZeroTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_batch(1, basis);
        auto actual = expected;

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::vector_set_constant(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_vector_batch(device_actual, basis),
            basis,
            Helper::tensor_count,
            Helper::Scalar{0});
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
            Helper::launch_cpu([&](auto const& strategy, auto config) {
                return rpp::ops::vector_set_constant(
                    strategy,
                    std::move(config),
                    Helper::host_vector_batch(expected, basis),
                    basis,
                    Helper::tensor_count,
                    Helper::Scalar{0});
            });
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-5});
    }
}

} // namespace
