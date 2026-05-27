#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_assign.hpp>
#include <rpp/gpu/block/operations/linalg/vector_assign.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockVectorAssignTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_batch(1, basis);
        auto actual = expected;
        auto const arg = Helper::make_batch(2, basis);

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::vector_assign(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_vector_batch(device_actual, basis),
            Helper::device_vector_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
             rpp::ops::vector_assign(cpu_strategy,
                                    Helper::CpuStrategy::LaunchConfig{},
                    Helper::host_vector_batch(expected, basis),
                    Helper::host_vector_batch(arg, basis),
                    basis,
                    Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-5});
    }
}

} // namespace
