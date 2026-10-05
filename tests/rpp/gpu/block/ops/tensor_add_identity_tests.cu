#include <gtest/gtest.h>

#include <limits>

#include "../../../tensor_identity_test_helper.hpp"

#include <rpp/gpu/block/operations/basic/tensor_add_identity.hpp>

#include "gpu_typed_adjoint_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockTensorAddIdentityTypedTests
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

    static HostVector reference_add_identity(HostVector const& initial,
                                             DegreeRange range,
                                             Accum scalar) {
        auto result = initial;
        if (range.min == 0) {
            auto const updated =
                static_cast<Accum>(result[0]) + static_cast<Accum>(scalar);
            result[0] = scalar_from_accum(updated);
        }
        return result;
    }

    static HostVector run_gpu_add_identity(Basis const& basis,
                                           GpuStrategy const& gpu_strategy,
                                           HostVector const& initial,
                                           DegreeRange range,
                                           Accum scalar,
                     typename Base::Index storage_offset = 0) {
        DeviceVector device_actual(initial);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_add_identity(
            gpu_strategy,
            std::move(launch_config),
            rpp::make_tensor_batch(Helper::device_data(device_actual) + storage_offset,
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

TYPED_TEST_SUITE(GpuBlockTensorAddIdentityTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockTensorAddIdentityTypedTests,
           AddsToIdentityCoefficientOnFullView) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{-2.5};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(4, basis);
        auto const range = typename TestFixture::DegreeRange{0, basis.depth};

        auto const actual = TestFixture::run_gpu_add_identity(
            basis, gpu_strategy, initial, range, scalar);
        auto const expected = TestFixture::reference_add_identity(
            initial, range, scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockTensorAddIdentityTypedTests,
           IsNoOpWhenViewExcludesIdentity) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{3.0};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        if (basis.depth < 1) {
            continue;
        }
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(5, basis);
        auto const range = typename TestFixture::DegreeRange{1, basis.depth};

        auto const actual = TestFixture::run_gpu_add_identity(
            basis, gpu_strategy, initial, range, scalar);
        auto const expected = TestFixture::reference_add_identity(
            initial, range, scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockTensorAddIdentityTypedTests,
           UpdatesIdentityOnlyWhenViewIncludesDegreeZero) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const scalar = typename TestFixture::Accum{0.875};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data =
            typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy =
            typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
        auto const initial = TestFixture::make_batch(6, basis);
        auto const max_degree =
            static_cast<typename TestFixture::Degree>(
                std::min<typename TestFixture::Degree>(1, basis.depth));
        auto const range = typename TestFixture::DegreeRange{0, max_degree};

        auto const actual = TestFixture::run_gpu_add_identity(
            basis, gpu_strategy, initial, range, scalar);
        auto const expected = TestFixture::reference_add_identity(
            initial, range, scalar);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockTensorAddIdentityTypedTests, AddIdentityPreservesAllOtherCoefficients) {
    RPP_REQUIRE_CUDA_DEVICE();
    using Range = typename TestFixture::DegreeRange;
    using Degree = typename TestFixture::Degree;
    using Index = typename TestFixture::Index;
    using Scalar = typename TestFixture::Scalar;
    using Accum = typename TestFixture::Accum;
    struct Config { Degree width, depth; };
    Config const configs[] = {{1, 4}, {4, 4}, {4, 0}};
    for (auto const& config : configs) {
        SCOPED_TRACE(testing::Message() << "width=" << config.width
                                       << ", depth=" << config.depth);
        auto const basis_data = typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};
        auto backing = TestFixture::make_batch(101, basis);
        backing.resize(static_cast<std::size_t>(basis.size()) + 2);
        for (std::size_t i = 0; i < backing.size(); ++i) {
            backing[i] = static_cast<Scalar>(static_cast<float>((i * 3) % 15 + 1) / 8.0f);
        }
        backing.front() = static_cast<Scalar>(7.0f);
        backing.back() = static_cast<Scalar>(-7.0f);
        auto const ranges = rpp::tests::all_tensor_degree_ranges<Range>(basis.depth);
        for (auto const range : ranges) {
            SCOPED_TRACE(testing::Message() << "range=[" << range.min << ',' << range.max << ']');
            for (auto const scalar : {Accum{0}, Accum{1}, Accum{-0.5}}) {
                SCOPED_TRACE(static_cast<double>(scalar));
                auto initial = backing;
                auto const expected = rpp::tests::reference_tensor_identity(
                    initial, basis, range, scalar, false, Index{1});
                auto const actual = TestFixture::run_gpu_add_identity(
                    basis, strategy, initial, range, scalar, Index{1});
                ASSERT_EQ(actual.size(), expected.size());
                for (std::size_t i = 0; i < expected.size(); ++i) {
                    // Dyadic values permit exact checks, including untouched guards.
                    EXPECT_EQ(static_cast<double>(actual[i]),
                              static_cast<double>(expected[i])) << "coefficient " << i;
                }
            }
        }
    }
}

} // namespace
