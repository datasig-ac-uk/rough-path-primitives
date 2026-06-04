#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/basic/ft_inplace_fma.hpp>

#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockFtInplaceFmaTypedTests
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

    template <rpp::ops::FTInplaceFMAType FmaType>
    static HostVector run_gpu_inplace_fma(Basis const& basis,
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

        auto b_batch = Helper::device_tensor_batch(device_b, basis);
        auto c_batch = Helper::device_tensor_batch(device_c, basis);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_inplace_fma<FmaType>(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_tensor_batch(
                Helper::device_data(device_actual), basis.size(), a_range.min,
                a_range.max),
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

TYPED_TEST_SUITE(GpuBlockFtInplaceFmaTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockFtInplaceFmaTypedTests,
           AEqualsBCPlusAMatchesOutOfPlaceReference) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{0.5};
    auto const beta = typename TestFixture::Accum{-1.25};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const a = TestFixture::make_batch(1, basis);
        auto const b = TestFixture::make_batch(2, basis);
        auto const c = TestFixture::make_batch(3, basis);

        auto const actual =
            TestFixture::template run_gpu_inplace_fma<
                rpp::ops::FTInplaceFMAType::AEqualsBCPlusA>(
                basis, gpu_strategy, a, b, c, TestFixture::full_range(basis),
                TestFixture::full_range(basis), TestFixture::full_range(basis),
                alpha, beta);
        auto const expected =
            TestFixture::reference_fma(basis, a, b, c, alpha, beta);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockFtInplaceFmaTypedTests,
           AEqualsBCPlusAMatchesOutOfPlaceReferenceForViews) {
    RPP_REQUIRE_CUDA_DEVICE();

    using Range = typename TestFixture::DegreeRange;

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const a = TestFixture::make_batch(4, basis);
        auto const b = TestFixture::make_batch(5, basis);
        auto const c = TestFixture::make_batch(6, basis);
        auto const a_range = Range{1, std::min<typename TestFixture::Degree>(
                                          3, basis.depth)};
        auto const b_range = Range{0, std::min<typename TestFixture::Degree>(
                                          2, basis.depth)};
        auto const c_range = Range{1, basis.depth};

        auto const actual =
            TestFixture::template run_gpu_inplace_fma<
                rpp::ops::FTInplaceFMAType::AEqualsBCPlusA>(
                basis, gpu_strategy, a, b, c, a_range, b_range, c_range);
        auto expected = a;
        auto const updated = TestFixture::reference_fma(
            basis, a, b, c, a_range, a_range, b_range, c_range);
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
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockFtInplaceFmaTypedTests,
           OrderedVariantsMatchOutOfPlaceReference) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{-0.5};
    auto const beta = typename TestFixture::Accum{0.75};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const initial_a = TestFixture::make_batch(7, basis);
        auto const b = TestFixture::make_batch(8, basis);
        auto const c = TestFixture::make_batch(9, basis);

        auto const actual_ab =
            TestFixture::template run_gpu_inplace_fma<
                rpp::ops::FTInplaceFMAType::AEqualsABPlusC>(
                basis, gpu_strategy, initial_a, b, c,
                TestFixture::full_range(basis), TestFixture::full_range(basis),
                TestFixture::full_range(basis), alpha, beta);
        auto const expected_ab =
            TestFixture::reference_fma(basis, c, initial_a, b, alpha, beta);
        TestFixture::expect_tensor_near(actual_ab, expected_ab);

        auto const actual_ba =
            TestFixture::template run_gpu_inplace_fma<
                rpp::ops::FTInplaceFMAType::AEqualsBAPlusC>(
                basis, gpu_strategy, initial_a, b, c,
                TestFixture::full_range(basis), TestFixture::full_range(basis),
                TestFixture::full_range(basis), alpha, beta);
        auto const expected_ba =
            TestFixture::reference_fma(basis, c, b, initial_a, alpha, beta);
        TestFixture::expect_tensor_near(actual_ba, expected_ba);
    }
}

} // namespace
