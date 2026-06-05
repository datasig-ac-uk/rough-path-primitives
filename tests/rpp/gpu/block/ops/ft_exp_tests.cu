#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/intermediate/ft_exp.hpp>
#include <rpp/gpu/block/operations/intermediate/ft_log.hpp>

#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockFtExpTypedTests
    : public rpp::tests::TypedGpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::expect_tensor_near;
    using Base::make_batch;
    using Base::make_unit_tensor;
    using Base::make_zero_batch;

    static HostVector make_positive_degree_tensor(Basis const& basis,
                                                  unsigned seed) {
        auto result = make_batch(seed, basis);
        result[0] = rpp::tests::cast_scalar<typename Base::Scalar>(0.0f);
        return result;
    }

    static HostVector run_gpu_exp(Basis const& basis,
                                  GpuStrategy const& gpu_strategy,
                                  HostVector const& arg,
                                  typename Base::DegreeRange arg_range) {
        auto actual = make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_exp(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            rpp::make_tensor_batch(
                Helper::device_data(device_arg), basis.size(), arg_range.min, arg_range.max),
            basis,
            Helper::tensor_count);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return actual;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return actual;
        }

        return Helper::copy_to_host(device_actual);
    }

    static HostVector run_gpu_log(Basis const& basis,
                                  GpuStrategy const& gpu_strategy,
                                  HostVector const& arg) {
        auto actual = make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_log(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return actual;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return actual;
        }

        return Helper::copy_to_host(device_actual);
    }
};

TYPED_TEST_SUITE(GpuBlockFtExpTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockFtExpTypedTests, LogExpRoundTripForPositiveDegreeInput) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const x = TestFixture::make_positive_degree_tensor(basis, 1);
        auto const exp_x = TestFixture::run_gpu_exp(
            basis, gpu_strategy, x, {1, basis.depth});
        auto const log_exp_x = TestFixture::run_gpu_log(basis, gpu_strategy, exp_x);

        TestFixture::expect_tensor_near(log_exp_x, x);
    }
}

TYPED_TEST(GpuBlockFtExpTypedTests, ExpOfZeroIsIdentity) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const actual = TestFixture::run_gpu_exp(
            basis, gpu_strategy, TestFixture::make_zero_batch(basis), {0, basis.depth});
        auto const expected = TestFixture::make_unit_tensor(basis);

        TestFixture::expect_tensor_near(actual, expected);
    }
}

} // namespace
