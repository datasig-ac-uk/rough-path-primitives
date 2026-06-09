#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/st_inplace_fma.hpp>
#include <rpp/gpu/block/operations/basic/st_inplace_fma.hpp>

#include "gpu_block_test_helper.cuh"
#include "gpu_typed_st_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockStInplaceFmaTypedTests
    : public rpp::tests::TypedGpuShuffleTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuShuffleTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DegreeRange;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_batch;
    using Base::make_zero_batch;
    using Base::reference_fma;

    [[nodiscard]] static HostVector run_gpu_inplace_fma(
        Basis const& basis,
        GpuStrategy const& gpu_strategy,
        HostVector const& initial_a,
        HostVector const& b,
        HostVector const& c,
        DegreeRange a_range,
        DegreeRange b_range,
        DegreeRange c_range,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}) {
        auto actual = initial_a;

        DeviceVector device_actual(actual);
        DeviceVector device_b(b);
        DeviceVector device_c(c);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::st_inplace_fma(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_tensor_batch(Helper::device_data(device_actual),
                                   basis.size(),
                                   a_range.min,
                                   a_range.max),
            rpp::make_tensor_batch(Helper::device_data(device_b),
                                   basis.size(),
                                   b_range.min,
                                   b_range.max),
            rpp::make_tensor_batch(Helper::device_data(device_c),
                                   basis.size(),
                                   c_range.min,
                                   c_range.max),
            basis,
            Helper::tensor_count,
            alpha,
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

TYPED_TEST_SUITE(GpuBlockStInplaceFmaTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockStInplaceFmaTypedTests,
           MatchesOutOfPlaceReferenceOnFullView) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const alpha = typename TestFixture::Accum{0.75};
        auto const beta = typename TestFixture::Accum{-1.25};

        auto const initial_a = TestFixture::make_batch(1, basis);
        auto const b = TestFixture::make_batch(2, basis);
        auto const c = TestFixture::make_batch(3, basis);

        auto const actual = TestFixture::run_gpu_inplace_fma(
            basis,
            gpu_strategy,
            initial_a,
            b,
            c,
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            alpha,
            beta);
        auto const expected = TestFixture::reference_fma(
            basis,
            initial_a,
            b,
            c,
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            TestFixture::full_range(basis),
            alpha,
            beta);

        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockStInplaceFmaTypedTests,
           MatchesOutOfPlaceReferenceForTruncatedViews) {
    RPP_REQUIRE_CUDA_DEVICE();

    using Range = typename TestFixture::DegreeRange;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};

        auto const a_range = Range{1, std::min<typename TestFixture::Degree>(3, basis.depth)};
        auto const b_range = Range{1, std::min<typename TestFixture::Degree>(2, basis.depth)};
        auto const c_range = Range{1, basis.depth};

        auto const initial_a = TestFixture::make_batch(4, basis);
        auto const b = TestFixture::make_batch(5, basis);
        auto const c = TestFixture::make_batch(6, basis);

        auto const actual = TestFixture::run_gpu_inplace_fma(
            basis, gpu_strategy, initial_a, b, c, a_range, b_range, c_range);
        auto expected = initial_a;
        auto const updated = TestFixture::reference_fma(
            basis, initial_a, b, c, a_range, a_range, b_range, c_range);
        for (typename TestFixture::Degree degree = a_range.min;
             degree <= a_range.max;
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

TYPED_TEST(GpuBlockStInplaceFmaTypedTests,
           MatchesOutOfPlaceReferenceWithScaledAddendAndProduct) {
    RPP_REQUIRE_CUDA_DEVICE();

    using Range = typename TestFixture::DegreeRange;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const alpha = typename TestFixture::Accum{1.25};
        auto const beta = typename TestFixture::Accum{0.5};

        auto const a_range = Range{1, basis.depth};
        auto const b_range = Range{0, std::min<typename TestFixture::Degree>(2, basis.depth)};
        auto const c_range = Range{1, basis.depth};

        auto const initial_a = TestFixture::make_batch(7, basis);
        auto const b = TestFixture::make_batch(8, basis);
        auto const c = TestFixture::make_batch(9, basis);

        auto const actual = TestFixture::run_gpu_inplace_fma(
            basis, gpu_strategy, initial_a, b, c, a_range, b_range, c_range, alpha, beta);
        auto expected = initial_a;
        auto const updated = TestFixture::reference_fma(
            basis,
            initial_a,
            b,
            c,
            a_range,
            a_range,
            b_range,
            c_range,
            alpha,
            beta);
        for (typename TestFixture::Degree degree = a_range.min;
             degree <= a_range.max;
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

TEST(GpuBlockStInplaceFmaTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();
        auto constexpr alpha = Helper::Scalar{0.25};
        auto constexpr beta = Helper::Scalar{-1.25};

        auto expected = Helper::make_batch(1, basis, Helper::Scalar{0.01});
        auto actual = expected;
        auto const b = Helper::make_batch(2, basis, Helper::Scalar{0.01});
        auto const c = Helper::make_batch(3, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_b(b);
        Helper::DeviceVector<Helper::Scalar> device_c(c);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::st_inplace_fma(
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

        auto const cpu_err = rpp::ops::st_inplace_fma(
            cpu_strategy,
            Helper::CpuStrategy::LaunchConfig{},
            Helper::host_tensor_batch(expected, basis),
            Helper::host_tensor_batch(b, basis),
            Helper::host_tensor_batch(c, basis),
            basis,
            Helper::tensor_count,
            alpha,
            beta);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-4});
    }
}

} // namespace
