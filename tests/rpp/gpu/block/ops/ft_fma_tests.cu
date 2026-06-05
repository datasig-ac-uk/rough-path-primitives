#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/basic/ft_fma.hpp>

#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockFtFmaTypedTests
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
    using Base::reference_fma;
    using Base::reference_mul;

    static HostVector run_gpu_fma(Basis const& basis,
                                  GpuStrategy const& gpu_strategy,
                                  HostVector const& a,
                                  HostVector const& b,
                                  HostVector const& c,
                                  DegreeRange out_range,
                                  DegreeRange a_range,
                                  DegreeRange b_range,
                                  DegreeRange c_range,
                                  Accum alpha = Accum{1},
                                  Accum beta = Accum{1}) {
        auto actual = Base::make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_a(a);
        DeviceVector device_b(b);
        DeviceVector device_c(c);

        auto a_batch = Helper::device_tensor_batch(device_a, basis);
        auto b_batch = Helper::device_tensor_batch(device_b, basis);
        auto c_batch = Helper::device_tensor_batch(device_c, basis);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_fma(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_tensor_batch(
                Helper::device_data(device_actual), basis.size(), out_range.min,
                out_range.max),
            rpp::make_tensor_batch(
                a_batch.data(), a_batch.layout(), a_range.min, a_range.max),
            rpp::make_tensor_batch(
                b_batch.data(), b_batch.layout(), b_range.min, b_range.max),
            rpp::make_tensor_batch(
                c_batch.data(), c_batch.layout(), c_range.min, c_range.max),
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

TYPED_TEST_SUITE(GpuBlockFtFmaTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockFtFmaTypedTests, MatchesHostReferenceForSingleElementBatches) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{-0.25};
    auto const beta = typename TestFixture::Accum{1.5};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const a = TestFixture::make_batch(1, basis);
        auto const b = TestFixture::make_batch(2, basis);
        auto const c = TestFixture::make_batch(3, basis);

        auto const actual = TestFixture::run_gpu_fma(
            basis, gpu_strategy, a, b, c, TestFixture::full_range(basis),
            TestFixture::full_range(basis), TestFixture::full_range(basis),
            TestFixture::full_range(basis), alpha, beta);
        auto const expected =
            TestFixture::reference_fma(basis, a, b, c, alpha, beta);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockFtFmaTypedTests, MatchesMulThenAddReference) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{0.5};
    auto const beta = typename TestFixture::Accum{-1.25};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const a = TestFixture::make_batch(4, basis);
        auto const b = TestFixture::make_batch(6, basis);
        auto const c = TestFixture::make_batch(8, basis);

        auto const actual = TestFixture::run_gpu_fma(
            basis, gpu_strategy, a, b, c, TestFixture::full_range(basis),
            TestFixture::full_range(basis), TestFixture::full_range(basis),
            TestFixture::full_range(basis), alpha, beta);

        auto expected = TestFixture::reference_mul(basis, b, c, beta);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expected[i] = rpp::tests::cast_scalar<typename TestFixture::Scalar>(
                typename TestFixture::Accum{expected[i]} +
                alpha * typename TestFixture::Accum{a[i]});
        }
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockFtFmaTypedTests, RespectsTruncatedOperandAndOutputViews) {
    RPP_REQUIRE_CUDA_DEVICE();

    using Range = typename TestFixture::DegreeRange;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const a = TestFixture::make_batch(5, basis);
        auto const b = TestFixture::make_batch(7, basis);
        auto const c = TestFixture::make_batch(9, basis);
        auto const out_range = Range{1, std::min<typename TestFixture::Degree>(
                                            3, basis.depth)};
        auto const a_range = Range{1, basis.depth};
        auto const b_range = Range{0, std::min<typename TestFixture::Degree>(
                                          2, basis.depth)};
        auto const c_range = Range{1, basis.depth};

        auto const actual = TestFixture::run_gpu_fma(
            basis, gpu_strategy, a, b, c, out_range, a_range, b_range, c_range);
        auto const expected = TestFixture::reference_fma(
            basis, a, b, c, out_range, a_range, b_range, c_range);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

} // namespace
