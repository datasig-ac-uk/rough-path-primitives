#include <gtest/gtest.h>

#include "../../../tensor_antipode_test_helper.hpp"

#include <rpp/gpu/block/operations/basic/ft_mul.hpp>
#include <rpp/gpu/block/operations/basic/tensor_antipode.hpp>

#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockTensorAntipodeTypedTests
    : public rpp::tests::TypedGpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Degree;
    using typename Base::DeviceVector;
    using typename Base::GpuStrategy;
    using typename Base::Helper;
    using typename Base::HostVector;
    using Base::expect_tensor_near;
    using Base::make_unit_tensor;
    using Base::make_zero_batch;

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

    static HostVector make_signed_batch(Basis const& basis, unsigned seed) {
        auto result = make_zero_batch(basis);
        for (std::size_t i = 0; i < result.size(); ++i) {
            auto const centered =
                static_cast<int>((seed * 19u + static_cast<unsigned>(i) * 11u) %
                                     11u) -
                5;
            auto const value = static_cast<Accum>(centered) * Accum{0.03125};
            result[i] = scalar_from_accum(value);
        }
        return result;
    }

    static HostVector linear_combo(HostVector const& lhs,
                                   Accum lhs_scale,
                                   HostVector const& rhs,
                                   Accum rhs_scale) {
        HostVector result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = scalar_from_accum(lhs_scale * static_cast<Accum>(lhs[i]) +
                                          rhs_scale * static_cast<Accum>(rhs[i]));
        }
        return result;
    }

    static HostVector make_group_like_argument(Basis const& basis, unsigned seed) {
        auto group_like = make_zero_batch(basis);
        auto const lambda =
            static_cast<Accum>((static_cast<unsigned>(seed % 3u) + 1u) *
                               0.03125);

        group_like[0] = scalar_from_accum(Accum{1});
        Accum coeff{1};
        for (Degree degree = 1; degree <= basis.depth; ++degree) {
            coeff = coeff * lambda / static_cast<Accum>(degree);
            group_like[static_cast<std::size_t>(basis.start_of_degree(degree))] =
                scalar_from_accum(coeff);
        }
        return group_like;
    }

    static HostVector run_gpu_antipode(Basis const& basis,
                                       GpuStrategy const& gpu_strategy,
                                       HostVector const& arg) {
        auto actual = make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_antipode(
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
            ADD_FAILURE() << cudaGetErrorString(sync_err);
            return actual;
        }

        return Helper::copy_to_host(device_actual);
    }

    static HostVector run_gpu_mul(Basis const& basis,
                                  GpuStrategy const& gpu_strategy,
                                  HostVector const& lhs,
                                  HostVector const& rhs) {
        auto actual = make_zero_batch(basis);

        DeviceVector device_actual(actual);
        DeviceVector device_lhs(lhs);
        DeviceVector device_rhs(rhs);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::ft_mul(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_lhs, basis),
            Helper::device_tensor_batch(device_rhs, basis),
            basis,
            Helper::tensor_count);
        if (!static_cast<bool>(err)) {
            ADD_FAILURE() << err.message();
            return actual;
        }
        auto const sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            ADD_FAILURE() << cudaGetErrorString(sync_err);
            return actual;
        }

        return Helper::copy_to_host(device_actual);
    }
};

TYPED_TEST_SUITE(GpuBlockTensorAntipodeTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockTensorAntipodeTypedTests, IsLinear) {
    RPP_REQUIRE_CUDA_DEVICE();

    auto const alpha = typename TestFixture::Accum{0.5};
    auto const beta = typename TestFixture::Accum{-1.25};

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const lhs = TestFixture::make_signed_batch(basis, 1);
        auto const rhs = TestFixture::make_signed_batch(basis, 2);
        auto const arg = TestFixture::linear_combo(lhs, alpha, rhs, beta);

        auto const antipode_arg =
            TestFixture::run_gpu_antipode(basis, gpu_strategy, arg);
        auto const antipode_lhs =
            TestFixture::run_gpu_antipode(basis, gpu_strategy, lhs);
        auto const antipode_rhs =
            TestFixture::run_gpu_antipode(basis, gpu_strategy, rhs);

        auto const expected =
            TestFixture::linear_combo(antipode_lhs, alpha, antipode_rhs, beta);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, antipode_arg, expected);
    }
}

TYPED_TEST(GpuBlockTensorAntipodeTypedTests,
           GrouplikeElementHasAntipodeInverseUnderMul) {
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = typename TestFixture::Helper::BasisData(
            config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const gpu_strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};

        auto const arg = TestFixture::make_group_like_argument(basis, 7);
        auto const antipode =
            TestFixture::run_gpu_antipode(basis, gpu_strategy, arg);
        auto const product =
            TestFixture::run_gpu_mul(basis, gpu_strategy, antipode, arg);
        auto const expected = TestFixture::make_unit_tensor(basis);

        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, product, expected);
    }
}

TYPED_TEST(GpuBlockTensorAntipodeTypedTests, ZeroExtendsArgumentAndPreservesOutsideOutputView) {
    RPP_REQUIRE_CUDA_DEVICE();
    auto const basis_data = typename TestFixture::Helper::BasisData(3, 4);
    auto const gpu_strategy =
        typename TestFixture::GpuStrategy{TestFixture::Helper::block_size};
    using Scalar = typename TestFixture::Scalar;
    using Degree = typename TestFixture::Degree;
    using Index = typename TestFixture::Index;
    using DegreeRange = typename TestFixture::DegreeRange;
    auto const& basis = basis_data.basis;
    typename TestFixture::HostVector arg(
        basis.size(), rpp::tests::cast_scalar<Scalar>(0.0f));
    for (std::size_t i = 0; i < arg.size(); ++i) {
        arg[i] = rpp::tests::cast_scalar<Scalar>(static_cast<float>(i + 1));
    }
    typename TestFixture::HostVector initial_out(
        arg.size(), rpp::tests::cast_scalar<Scalar>(7.0f));

    struct Case {
        const char* name;
        DegreeRange out;
        DegreeRange arg;
    };
    const Case cases[] = {
        {"missing lower and upper degrees", {0, 4}, {2, 3}},
        {"single-degree overlap in truncated output", {1, 3}, {2, 2}},
        {"input wider than output", {1, 2}, {0, 4}},
        {"argument entirely above unit-only output", {0, 0}, {1, 4}},
        {"argument entirely below output", {2, 3}, {0, 1}},
        {"unit-only argument", {0, 4}, {0, 0}},
        {"unit-only input and output", {0, 0}, {0, 0}},
        {"matching truncated ranges", {1, 3}, {1, 3}},
    };
    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto actual = initial_out;
        auto expected = initial_out;
        for (Degree degree = test_case.out.min; degree <= test_case.out.max;
             ++degree) {
            const auto begin = basis.start_of_degree(degree);
            for (Index i = 0; i < basis.size_of_degree(degree); ++i) {
                const auto target = static_cast<std::size_t>(begin + i);
                if (degree < test_case.arg.min || degree > test_case.arg.max) {
                    expected[target] = rpp::tests::cast_scalar<Scalar>(0.0f);
                    continue;
                }

                // Reverse the base-width digits independently of reverse_index.
                Index source_index = i;
                Index reversed{0};
                for (Degree letter = 0; letter < degree; ++letter) {
                    reversed = reversed * basis.width + source_index % basis.width;
                    source_index /= basis.width;
                }
                const auto value = arg[static_cast<std::size_t>(begin + reversed)];
                expected[target] = degree % 2 == 0 ? value : -value;
            }
        }

        typename TestFixture::DeviceVector device_actual(actual);
        typename TestFixture::DeviceVector device_arg(arg);
        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_antipode(
            gpu_strategy,
            launch_config,
            rpp::make_tensor_batch(
                TestFixture::Helper::device_data(device_actual),
                basis.size(), test_case.out.min, test_case.out.max),
            rpp::make_tensor_batch(
                TestFixture::Helper::device_data(device_arg),
                basis.size(), test_case.arg.min, test_case.arg.max),
            basis,
            TestFixture::Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());
        actual = TestFixture::Helper::copy_to_host(device_actual);
        RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
    }
}

TYPED_TEST(GpuBlockTensorAntipodeTypedTests, HandlesEveryInputAndOutputDegreeRange) {
    RPP_REQUIRE_CUDA_DEVICE();
    using Range = typename TestFixture::DegreeRange;
    using Scalar = typename TestFixture::Scalar;
    using Degree = typename TestFixture::Degree;
    struct Config {
        Degree width;
        Degree depth;
    };
    Config const configs[] = {{1, 4}, {4, 4}, {4, 0}};
    for (auto const& config : configs) {
        SCOPED_TRACE(testing::Message() << "width=" << config.width
                                       << ", depth=" << config.depth);
        auto const basis_data = typename TestFixture::Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto initial_out = TestFixture::make_batch(71, basis);
        for (auto& value : initial_out) {
            value = static_cast<Scalar>(-7.0f);
        }
        auto const arg = rpp::tests::make_antipode_range_argument(
            TestFixture::make_batch(72, basis));
        auto const ranges = rpp::tests::all_tensor_degree_ranges<Range>(basis.depth);
        auto const strategy = typename TestFixture::GpuStrategy{
            TestFixture::Helper::block_size};
        typename TestFixture::DeviceVector device_arg(arg);
        for (auto const out_range : ranges) {
            for (auto const arg_range : ranges) {
                SCOPED_TRACE(testing::Message()
                             << "out=[" << out_range.min << ',' << out_range.max
                             << "], arg=[" << arg_range.min << ',' << arg_range.max << ']');
                auto const expected = rpp::tests::reference_tensor_antipode<
                    typename TestFixture::Accum>(
                    initial_out, arg, basis, out_range, arg_range,
                    true);
                typename TestFixture::DeviceVector device_actual(initial_out);
                rpp::gpu::DeviceLaunchConfig launch_config;
                launch_config.stream = nullptr;
                auto const err = rpp::ops::tensor_generalised_antipode<
                    rpp::ops::TensorAntipodeSigningPolicy::SignByDegree>(
                    strategy, launch_config,
                    rpp::make_tensor_batch(
                        TestFixture::Helper::device_data(device_actual), basis.size(),
                        out_range.min, out_range.max),
                    rpp::make_tensor_batch(
                        TestFixture::Helper::device_data(device_arg), basis.size(),
                        arg_range.min, arg_range.max),
                    basis, TestFixture::Helper::tensor_count);
                ASSERT_TRUE(static_cast<bool>(err)) << err.message();
                RPP_CUDA_ASSERT(cudaDeviceSynchronize());
                auto const actual = TestFixture::Helper::copy_to_host(device_actual);
                RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(TestFixture, actual, expected);
            }
        }
    }
}

} // namespace
