#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/intermediate/ft_exp.hpp>
#include <rpp/gpu/block/operations/intermediate/ft_fmexp.hpp>

#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockFtFmexpTypedTests
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
    using Base::reference_mul;

    static constexpr double multiply_by_exponential_tolerance() {
        if constexpr (std::is_same_v<typename Base::Scalar, double> &&
                      std::is_same_v<Accum, double>) {
            return 1e-8;
        }
        return rpp::tests::NumericTolerance<typename Base::Scalar, Accum>::value;
    }

    static void expect_tensor_near_for_multiply_by_exponential(
        HostVector const& actual, HostVector const& expected) {
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i) {
            auto const actual_d = rpp::tests::scalar_to_double(actual[i]);
            auto const expected_d = rpp::tests::scalar_to_double(expected[i]);
            auto const scale =
                std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
            auto const tolerance = multiply_by_exponential_tolerance() * scale;
            EXPECT_NEAR(actual_d, expected_d, tolerance)
                << "at coefficient " << i;
        }
    }

    static HostVector make_positive_degree_tensor(Basis const& basis,
                                                  unsigned seed) {
        auto result = make_batch(seed, basis);
        result[0] = rpp::tests::cast_scalar<typename Base::Scalar>(0.0f);
        return result;
    }

    static HostVector run_gpu_exp(Basis const& basis,
                                  GpuStrategy const& gpu_strategy,
                                  HostVector const& arg) {
        auto actual = make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_exp(
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

    static HostVector run_gpu_fmexp(Basis const& basis,
                                    GpuStrategy const& gpu_strategy,
                                    HostVector const& multiplier,
                                    HostVector const& exponent) {
        auto actual = make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_multiplier(multiplier);
        DeviceVector device_exponent(exponent);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_fmexp(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_multiplier, basis),
            Helper::device_tensor_batch(device_exponent, basis),
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

TYPED_TEST_SUITE(GpuBlockFtFmexpTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockFtFmexpTypedTests, MultipliesByExponentialOnTheRight) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const x = TestFixture::make_positive_degree_tensor(basis, 1);
        auto const y = TestFixture::make_positive_degree_tensor(basis, 2);
        auto const exp_x = TestFixture::run_gpu_exp(basis, gpu_strategy, x);
        auto const exp_y = TestFixture::run_gpu_exp(basis, gpu_strategy, y);

        auto const actual = TestFixture::run_gpu_fmexp(basis, gpu_strategy, exp_x, y);
        auto const expected = TestFixture::reference_mul(basis, exp_x, exp_y);

        TestFixture::expect_tensor_near_for_multiply_by_exponential(actual,
                                                                    expected);
    }
}

TYPED_TEST(GpuBlockFtFmexpTypedTests, IdentityMultiplierIsExp) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const x = TestFixture::make_positive_degree_tensor(basis, 3);
        auto const actual = TestFixture::run_gpu_fmexp(
            basis, gpu_strategy, TestFixture::make_unit_tensor(basis), x);
        auto const expected = TestFixture::run_gpu_exp(basis, gpu_strategy, x);

        TestFixture::expect_tensor_near(actual, expected);
    }
}

} // namespace
