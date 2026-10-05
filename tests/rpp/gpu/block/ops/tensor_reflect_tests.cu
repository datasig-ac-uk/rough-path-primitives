#include <gtest/gtest.h>

#include "../../../tensor_antipode_test_helper.hpp"

#include <rpp/cpu/single_thread/operations/basic/tensor_reflect.hpp>
#include <rpp/gpu/block/operations/basic/tensor_reflect.hpp>

#include "gpu_block_test_helper.cuh"
#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockTensorReflectTypedTests
    : public rpp::tests::TypedGpuFreeTensorOpTestBase<Config> {};

TYPED_TEST_SUITE(GpuBlockTensorReflectTypedTests,
                 rpp::tests::TypedGpuAdjointTestTypes,
                 rpp::tests::TypedScalarAccumNameGenerator);

TEST(GpuBlockTensorReflectTests, MatchesCpuForSingleElementBatches) {
    using Helper = rpp::tests::GpuBlockTestHelper;
    RPP_REQUIRE_CUDA_DEVICE();

    for (auto const& config : rpp::tests::gpu_block_test_configs) {
        auto const basis_data = Helper::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto const cpu_strategy = Helper::cpu_strategy();
        auto const gpu_strategy = Helper::gpu_strategy();

        auto expected = Helper::make_zero_batch(basis);
        auto actual = expected;
        auto const arg = Helper::make_batch(2, basis, Helper::Scalar{0.01});

        Helper::DeviceVector<Helper::Scalar> device_actual(actual);
        Helper::DeviceVector<Helper::Scalar> device_arg(arg);

        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_reflect(
            gpu_strategy,
            std::move(launch_config),
            Helper::device_tensor_batch(device_actual, basis),
            Helper::device_tensor_batch(device_arg, basis),
            basis,
            Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(err)) << err.message();
        RPP_CUDA_ASSERT(cudaDeviceSynchronize());

        auto const cpu_err =
             rpp::ops::tensor_reflect(cpu_strategy,
                                    Helper::CpuStrategy::LaunchConfig{},
                    Helper::host_tensor_batch(expected, basis),
                    Helper::host_tensor_batch(arg, basis),
                    basis,
                    Helper::tensor_count);
        ASSERT_TRUE(static_cast<bool>(cpu_err)) << cpu_err.message();

        actual = Helper::copy_to_host(device_actual);
        Helper::expect_near(actual, expected, Helper::Scalar{1.5e-5});
    }
}

TYPED_TEST(GpuBlockTensorReflectTypedTests, ZeroExtendsArgumentAndPreservesOutsideOutputView) {
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
                expected[target] = value;
            }
        }

        typename TestFixture::DeviceVector device_actual(actual);
        typename TestFixture::DeviceVector device_arg(arg);
        rpp::gpu::DeviceLaunchConfig launch_config;
        launch_config.stream = nullptr;
        auto const err = rpp::ops::tensor_reflect(
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

TYPED_TEST(GpuBlockTensorReflectTypedTests, HandlesEveryInputAndOutputDegreeRange) {
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
                    false);
                typename TestFixture::DeviceVector device_actual(initial_out);
                rpp::gpu::DeviceLaunchConfig launch_config;
                launch_config.stream = nullptr;
                auto const err = rpp::ops::tensor_generalised_antipode<
                    rpp::ops::TensorAntipodeSigningPolicy::NoSigning>(
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
