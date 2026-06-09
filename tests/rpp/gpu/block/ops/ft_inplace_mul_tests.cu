#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/basic/ft_inplace_mul.hpp>

#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockFtInplaceMulTypedTests
    : public rpp::tests::TypedGpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_batch;
    using Base::make_unit_tensor;
    using Base::reference_mul;

    static HostVector run_gpu_inplace_mul(Basis const& basis,
                                          GpuStrategy const& gpu_strategy,
                                          HostVector const& initial_lhs,
                                          HostVector const& rhs,
                                          DegreeRange lhs_range,
                                          DegreeRange rhs_range,
                                          Accum beta = Accum{1}) {
        auto actual = initial_lhs;

        DeviceVector device_actual(actual);
        DeviceVector device_rhs(rhs);
        auto rhs_batch = Helper::device_tensor_batch(device_rhs, basis);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_inplace_mul(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_tensor_batch(
                Helper::device_data(device_actual), basis.size(), lhs_range.min,
                lhs_range.max),
            rpp::make_tensor_batch(
                rhs_batch.data(), rhs_batch.layout(), rhs_range.min, rhs_range.max),
            basis,
            Helper::tensor_count,
            beta);
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

TYPED_TEST_SUITE(GpuBlockFtInplaceMulTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockFtInplaceMulTypedTests,
           MatchesOutOfPlaceReferenceForSingleElementBatches) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const beta = typename TestFixture::Accum{0.75};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const lhs = TestFixture::make_batch(1, basis);
        auto const rhs = TestFixture::make_batch(2, basis);

        auto const actual = TestFixture::run_gpu_inplace_mul(
            basis, gpu_strategy, lhs, rhs, TestFixture::full_range(basis),
            TestFixture::full_range(basis), beta);
        auto const expected = TestFixture::reference_mul(basis, lhs, rhs, beta);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockFtInplaceMulTypedTests, MatchesOutOfPlaceReferenceForViews) {
    RPP_REQUIRE_CUDA_DEVICE();

    using Range = typename TestFixture::DegreeRange;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const lhs = TestFixture::make_batch(3, basis);
        auto const rhs = TestFixture::make_batch(4, basis);
        auto const lhs_range = Range{1, std::min<typename TestFixture::Degree>(
                                            3, basis.depth)};
        auto const rhs_range = Range{1, basis.depth};

        auto const actual = TestFixture::run_gpu_inplace_mul(
            basis, gpu_strategy, lhs, rhs, lhs_range, rhs_range);
        auto expected = lhs;
        auto const updated = TestFixture::reference_mul(
            basis, lhs, rhs, lhs_range, lhs_range, rhs_range);
        for (typename TestFixture::Degree degree = lhs_range.min;
             degree <= lhs_range.max;
             ++degree) {
            auto const begin = basis.start_of_degree(degree);
            auto const end = basis.end_of_degree(degree);
            for (typename TestFixture::Index idx = begin; idx < end; ++idx) {
                expected[static_cast<std::size_t>(idx)] =
                    updated[static_cast<std::size_t>(idx)];
            }
        }
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockFtInplaceMulTypedTests,
           UnitRightIdentityReturnsScaledArgument) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const beta = typename TestFixture::Accum{-1.25};
    constexpr auto double_tolerance = 5e-8;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const lhs = TestFixture::make_batch(10, basis);
        auto const unit = TestFixture::make_unit_tensor(basis);
        auto const actual = TestFixture::run_gpu_inplace_mul(
            basis, gpu_strategy, lhs, unit, TestFixture::full_range(basis),
            TestFixture::full_range(basis), beta);

        auto expected = lhs;
        for (auto& coeff : expected) {
            coeff = rpp::tests::cast_scalar<typename TestFixture::Scalar>(
                beta * typename TestFixture::Accum{coeff});
        }
        if constexpr (std::is_same_v<typename TestFixture::Scalar, double> &&
                      std::is_same_v<typename TestFixture::Accum, double>) {
            RPP_EXPECT_GPU_TYPED_TENSOR_NEAR_WITH_BASE_TOLERANCE(
                TestFixture, actual, expected, double_tolerance);
        }
        else {
            RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
        }
    }
}

TYPED_TEST(GpuBlockFtInplaceMulTypedTests,
           MatchesOutOfPlaceReferenceWithSmallBlockSizeOnW4D3) {
    RPP_REQUIRE_CUDA_DEVICE();

    using SmallBlockStrategy = rpp::gpu::strategies::BlockStrategy<
        typename TestFixture::Accum,
        32,
        256,
        typename TestFixture::Helper::GpuArchitecture>;

    auto const basis_data = typename TestFixture::Helper::BasisData(4, 3);
    auto const& basis = basis_data.basis;
    auto const gpu_strategy = SmallBlockStrategy{32};
    auto const beta = typename TestFixture::Accum{0.75};

    auto const lhs = TestFixture::make_batch(30, basis);
    auto const rhs = TestFixture::make_batch(31, basis);

    auto actual = lhs;

    typename TestFixture::DeviceVector device_actual(actual);
    typename TestFixture::DeviceVector device_rhs(rhs);
    auto rhs_batch = TestFixture::Helper::device_tensor_batch(device_rhs, basis);

    rpp::gpu::DeviceLaunchConfig launch_config;
    launch_config.stream = nullptr;
    auto const err = rpp::ops::ft_inplace_mul(
        gpu_strategy,
        std::move(launch_config),
        rpp::make_tensor_batch(TestFixture::Helper::device_data(device_actual),
                               basis.size(),
                               0,
                               basis.depth),
        rpp::make_tensor_batch(rhs_batch.data(),
                               rhs_batch.layout(),
                               0,
                               basis.depth),
        basis,
        TestFixture::Helper::tensor_count,
        beta);
    ASSERT_TRUE(static_cast<bool>(err)) << err.message();
    RPP_CUDA_ASSERT(cudaDeviceSynchronize());

    actual = TestFixture::Helper::copy_to_host(device_actual);
    auto const expected = TestFixture::reference_mul(basis, lhs, rhs, beta);
    RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
}

} // namespace
