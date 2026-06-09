#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/basic/tensor_set_identity.hpp>

#include "gpu_typed_adjoint_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockTensorSetIdentityTypedTests
    : public rpp::tests::TypedGpuAdjointTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuAdjointTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::expect_tensor_near;
    using Base::make_batch;
    using Base::make_zero_batch;

    struct DegreeRange {
        Degree min;
        Degree max;
    };

    static typename Base::Scalar scalar_from_accum(Accum value) {
        if constexpr (std::is_same_v<typename Base::Scalar, __half> ||
                      std::is_same_v<typename Base::Scalar, __nv_bfloat16>) {
            return rpp::tests::cast_scalar<typename Base::Scalar>(
                static_cast<float>(value));
        }
        else {
            return static_cast<typename Base::Scalar>(value);
        }
    }

    static HostVector reference_set_identity(HostVector const& initial,
                                             Basis const& basis,
                                             DegreeRange range,
                                             Accum scalar) {
        auto result = initial;
        for (auto idx = basis.start_of_degree(range.min);
             idx < basis.end_of_degree(range.max);
             ++idx) {
            result[static_cast<std::size_t>(idx)] = scalar_from_accum(Accum{0});
        }
        if (range.min == 0) {
            result[0] = scalar_from_accum(scalar);
        }
        return result;
    }

    static HostVector run_gpu_set_identity(Basis const& basis,
                                           GpuStrategy const& gpu_strategy,
                                           HostVector const& initial,
                                           DegreeRange range,
                                           Accum scalar) {
        DeviceVector device_actual(initial);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_set_identity(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_tensor_batch(Helper::device_data(device_actual),
                                   basis.size(),
                                   range.min,
                                   range.max),
            basis,
            Helper::tensor_count,
            scalar);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return make_zero_batch(basis);
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << "cudaDeviceSynchronize failed: "
                          << cudaGetErrorString(sync_err);
            return make_zero_batch(basis);
        }

        return Helper::copy_to_host(device_actual);
    }
};

TYPED_TEST_SUITE(GpuBlockTensorSetIdentityTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockTensorSetIdentityTypedTests, SetsUnitTensorOnFullView) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{4.75};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(1, basis);
        auto const range = typename TestFixture::DegreeRange{0, basis.depth};

        auto const actual = TestFixture::run_gpu_set_identity(
            basis, gpu_strategy, initial, range, scalar);
        auto const expected = TestFixture::reference_set_identity(
            initial, basis, range, scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockTensorSetIdentityTypedTests,
           ZerosOnlyActiveSliceWhenViewExcludesIdentity) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{-2.5};

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

        auto const actual = TestFixture::run_gpu_set_identity(
            basis, gpu_strategy, initial, range, scalar);
        auto const expected = TestFixture::reference_set_identity(
            initial, basis, range, scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockTensorSetIdentityTypedTests,
           SetsIdentityAndZerosOnlyActiveSliceWhenViewIncludesDegreeZero) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{1.25};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(3, basis);
        auto const max_degree =
            static_cast<typename TestFixture::Degree>(
                std::min<typename TestFixture::Degree>(1, basis.depth));
        auto const range = typename TestFixture::DegreeRange{0, max_degree};

        auto const actual = TestFixture::run_gpu_set_identity(
            basis, gpu_strategy, initial, range, scalar);
        auto const expected = TestFixture::reference_set_identity(
            initial, basis, range, scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

} // namespace
