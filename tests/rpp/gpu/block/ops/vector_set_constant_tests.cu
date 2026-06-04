#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>

#include "gpu_typed_vector_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockVectorSetConstantTypedTests
    : public rpp::tests::TypedGpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuVectorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::assign_slice;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_batch;
    using Base::scalar_from_accum;

    static HostVector run_gpu_set_constant(Basis const& basis,
                                           GpuStrategy const& gpu_strategy,
                                           HostVector const& initial,
                                           typename Base::DegreeRange range,
                                           Accum value) {
        DeviceVector device_actual(initial);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::vector_set_constant(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_graded_vector_batch(Helper::device_data(device_actual),
                                          basis.size(),
                                          basis,
                                          range.min,
                                          range.max),
            basis,
            Helper::tensor_count,
            value);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return initial;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return initial;
        }

        return Helper::copy_to_host(device_actual);
    }
};

TYPED_TEST_SUITE(GpuBlockVectorSetConstantTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockVectorSetConstantTypedTests, SetsFullViewToConstant) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const value = typename TestFixture::Accum{3.25};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(1, basis);

        auto const actual = TestFixture::run_gpu_set_constant(
            basis, gpu_strategy, initial, TestFixture::full_range(basis), value);
        auto const expected = TestFixture::assign_slice(
            initial, basis, TestFixture::full_range(basis),
            TestFixture::scalar_from_accum(value));
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockVectorSetConstantTypedTests, SetsOnlyActiveSliceForView) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const value = typename TestFixture::Accum{-1.5};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        if (basis.depth < 1) {
            continue;
        }
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(2, basis);
        auto const range = typename TestFixture::DegreeRange{1, basis.depth};

        auto const actual =
            TestFixture::run_gpu_set_constant(basis, gpu_strategy, initial, range, value);
        auto const expected = TestFixture::assign_slice(
            initial, basis, range, TestFixture::scalar_from_accum(value));
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(GpuBlockVectorSetConstantTypedTests, SettingZeroZerosActiveSlice) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(3, basis);
        auto const max_degree =
            static_cast<typename TestFixture::Degree>(std::min<int>(basis.depth, 1));
        auto const range = typename TestFixture::DegreeRange{0, max_degree};

        auto const actual = TestFixture::run_gpu_set_constant(
            basis, gpu_strategy, initial, range, typename TestFixture::Accum{0});
        auto const expected = TestFixture::assign_slice(
            initial, basis, range,
            TestFixture::scalar_from_accum(typename TestFixture::Accum{0}));
        TestFixture::expect_tensor_near(actual, expected);
    }
}

} // namespace
