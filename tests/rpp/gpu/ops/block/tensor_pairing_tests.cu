#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/tensor_pairing.hpp>
#include <rpp/gpu/operations/block/basic/tensor_pairing.hpp>

#include "gpu_block_test_helper.cuh"

namespace {


TEST(GpuBlockTensorPairingTests, MatchesCpuForSingleElementBatches)
{
    using Helper = rpp::tests::GpuBlockTestHelper;
    using GpuOp = rpp::ops::TensorPairing<Helper::GpuStrategy>;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto const functional_data = Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto const arg_data = Helper::make_batch(2, basis, Helper::Scalar{0.01});
        Helper::Scalar expected = 0;

        auto const cpu_ctx = Helper::CpuStrategy::make_context(nullptr);
        auto functional = Helper::host_tensor_batch(functional_data, basis).view(0, basis);
        auto arg = Helper::host_tensor_batch(arg_data, basis).view(0, basis);
        rpp::ops::TensorPairing<Helper::CpuStrategy>{}(cpu_ctx, expected, functional, arg);

        Helper::DeviceVector<Helper::Scalar> device_functional(functional_data);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg_data);
        Helper::DeviceVector<Helper::Scalar> device_actual(1);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_pairing(
            gpu_strategy,
            std::move(launch_config),
            rpp::dense::ScalarBatch(rpp::tag_pointer<typename Helper::GpuArchitecture>(Helper::device_data(device_actual))),
            Helper::device_tensor_batch(device_functional, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            1
        );
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const actual = Helper::copy_to_host(device_actual);
        ASSERT_EQ(actual.size(), std::size_t{1});
        Helper::expect_near(actual[0], expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
