#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/tensor_reflect.hpp>
#include <rpp/gpu/operations/block/basic/tensor_reflect.hpp>

#include "gpu_block_test_helper.cuh"

namespace {

TEST(GpuBlockTensorReflectTests, MatchesCpuForSingleElementBatches)
{
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_zero_batch(basis);
        auto actual = expected;
        auto const arg = Helper::make_batch(2, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);
        Helper::DeviceBasis device_basis(basis_data);

        rpp::gpu::block::tensor_reflect_kernel<<<
            Helper::tensor_count,
            gpu_strategy.block_size,
            0
        >>>(
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_arg, basis),
            device_basis.basis,
            gpu_strategy,
            Helper::tensor_count
        );
        RPP_CUDA_ASSERT(cudaGetLastError());
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        rpp::cpu::single_thread::tensor_reflect_kernel(
            Helper::host_tensor_batch(expected, basis),
            Helper::host_tensor_batch(arg, basis),
            basis,
            cpu_strategy,
            Helper::tensor_count
        );

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-5});
    }
}

} // namespace
