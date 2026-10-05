#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <rpp/gpu/block/operations/intermediate/ft_exp.hpp>
#include <rpp/gpu/block/operations/intermediate/ft_log.hpp>
#include <rpp/gpu/block/operations/intermediate/ft_fmexp.hpp>

#include "../../../ft_intermediate_test_helper.hpp"
#include "gpu_typed_ft_ops_test_helper.cuh"

namespace {

template <typename Config>
class GpuBlockIntermediateInitializationTests : public rpp::tests::TypedGpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedGpuFreeTensorOpTestBase<Config>;
    using typename Base::Scalar;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Index;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::Helper;
    using typename Base::GpuStrategy;
    using typename Base::DeviceVector;

    template <rpp::tests::IntermediateOperation Operation>
    static void check_output_initialization() {
        using Kind = rpp::tests::IntermediateOperation;
        struct BasisConfig { Degree width, depth; };
        BasisConfig const configs[] = {{1, 4}, {4, 4}, {4, 0}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                           << ", depth=" << config.depth);
            auto const basis_data = typename Helper::BasisData(config.width, config.depth);
            auto const& basis = basis_data.basis;
            for (auto const& ranges : rpp::tests::intermediate_range_cases<DegreeRange>(Operation, basis.depth)) {
                if (ranges.multiplier.min > ranges.multiplier.max || ranges.multiplier.max > basis.depth ||
                    ranges.arg.min > ranges.arg.max || ranges.arg.max > basis.depth) {
                    continue;
                }
                SCOPED_TRACE(ranges.name);
                auto const multiplier = rpp::tests::make_intermediate_input(
                    Base::make_batch(111, basis), basis, true, ranges.zero_multiplier);
                auto arg = rpp::tests::make_intermediate_input(
                    Base::make_batch(112, basis), basis, false, ranges.zero_arg);
                if constexpr (Operation == Kind::Log) {
                    arg[0] = static_cast<Scalar>(1.0f);
                }
                auto const coefficients = rpp::tests::reference_intermediate(
                    basis, multiplier, arg, ranges.multiplier, ranges.arg, Operation);
                // These Horner schemes require degree zero in the output view.
                // Test full outputs and shorter prefixes, without removing it.
                std::vector<Degree> output_maxima{0, std::min(Degree{2}, basis.depth), basis.depth};
                std::sort(output_maxima.begin(), output_maxima.end());
                output_maxima.erase(std::unique(output_maxima.begin(), output_maxima.end()), output_maxima.end());
                for (auto const output_max : output_maxima) {
                    SCOPED_TRACE(output_max);
                    auto initial = Base::make_batch(113, basis);
                    initial.resize(static_cast<std::size_t>(basis.size()) + 2,
                                   static_cast<Scalar>(7.0f));
                    for (auto& value : initial) {
                        value = static_cast<Scalar>(7.0f);
                    }
                    initial.back() = static_cast<Scalar>(-7.0f);
                    auto const output_end = basis.end_of_degree(output_max);
                    for (Index idx = 0; idx < output_end; ++idx) {
                        initial[idx + 1] = static_cast<Scalar>(std::numeric_limits<float>::quiet_NaN());
                    }
                    auto expected = initial;
                    for (Index idx = 0; idx < output_end; ++idx) {
                        expected[idx + 1] = static_cast<Scalar>(coefficients[idx]);
                    }
                    DeviceVector device_actual(initial), device_multiplier(multiplier), device_arg(arg);
                    auto const strategy = GpuStrategy{Helper::block_size};
                    rpp::gpu::DeviceLaunchConfig launch;
                    launch.stream = nullptr;
                    auto const out_batch = rpp::make_tensor_batch(
                        Helper::device_data(device_actual) + 1, basis.size(), Degree{0}, output_max);
                    auto const arg_batch = rpp::make_tensor_batch(
                        Helper::device_data(device_arg), basis.size(), ranges.arg.min, ranges.arg.max);
                    auto const multiplier_batch = rpp::make_tensor_batch(
                        Helper::device_data(device_multiplier), basis.size(), ranges.multiplier.min, ranges.multiplier.max);
                    auto const err = [&] {
                        if constexpr (Operation == Kind::Exp) {
                            return rpp::ops::ft_exp(strategy, launch, out_batch, arg_batch, basis, Index{1});
                        }
                        else if constexpr (Operation == Kind::Log) {
                            return rpp::ops::ft_log(strategy, launch, out_batch, arg_batch, basis, Index{1});
                        }
                        else {
                            return rpp::ops::ft_fmexp(strategy, launch, out_batch, multiplier_batch,
                                                     arg_batch, basis, Index{1});
                        }
                    }();
                    ASSERT_TRUE(static_cast<bool>(err)) << err.message();
                    RPP_CUDA_ASSERT(cudaDeviceSynchronize());
                    auto const actual = Helper::copy_to_host(device_actual);
                    ASSERT_EQ(actual.size(), expected.size());
                    for (Index idx = 0; idx < output_end; ++idx) {
                        EXPECT_TRUE(std::isfinite(static_cast<double>(actual[idx + 1])))
                            << "coefficient " << idx;
                        if (coefficients[idx] == 0.0) {
                            EXPECT_EQ(static_cast<double>(actual[idx + 1]), 0.0)
                                << "coefficient " << idx;
                        }
                    }
                    EXPECT_EQ(static_cast<double>(actual.front()), 7.0);
                    for (std::size_t i = static_cast<std::size_t>(output_end + 1); i < actual.size(); ++i) {
                        EXPECT_EQ(static_cast<double>(actual[i]), static_cast<double>(initial[i]))
                            << "outside output view at " << i;
                    }
                    RPP_EXPECT_GPU_TYPED_TENSOR_NEAR(GpuBlockIntermediateInitializationTests, actual, expected);
                }
            }
        }
    }
};

TYPED_TEST_SUITE(GpuBlockIntermediateInitializationTests, rpp::tests::TypedGpuAdjointTestTypes, rpp::tests::TypedScalarAccumNameGenerator);

TYPED_TEST(GpuBlockIntermediateInitializationTests, ExpInitializesOutputAndZeroExtendsOperands) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_output_initialization<rpp::tests::IntermediateOperation::Exp>();
}

TYPED_TEST(GpuBlockIntermediateInitializationTests, LogInitializesOutputAndZeroExtendsOperands) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_output_initialization<rpp::tests::IntermediateOperation::Log>();
}

TYPED_TEST(GpuBlockIntermediateInitializationTests, FMExpInitializesOutputAndZeroExtendsOperands) {
    RPP_REQUIRE_CUDA_DEVICE();
    TestFixture::template check_output_initialization<rpp::tests::IntermediateOperation::FMExp>();
}

} // namespace
