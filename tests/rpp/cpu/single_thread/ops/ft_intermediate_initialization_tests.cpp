#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/intermediate/ft_exp.hpp>
#include <rpp/cpu/single_thread/operations/intermediate/ft_log.hpp>
#include <rpp/cpu/single_thread/operations/intermediate/ft_fmexp.hpp>

#include "../../../ft_intermediate_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"

namespace {

template <typename Config>
class CpuIntermediateInitializationTests : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Scalar;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::Index;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::Strategy;

    template <rpp::tests::IntermediateOperation Operation>
    static void check_output_initialization() {
        using Kind = rpp::tests::IntermediateOperation;
        struct BasisConfig { Degree width, depth; };
        BasisConfig const configs[] = {{1, 4}, {4, 4}, {4, 0}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                           << ", depth=" << config.depth);
            auto const basis_data = typename Base::BasisData(config.width, config.depth);
            auto const& basis = basis_data.basis;
            for (auto const& ranges : rpp::tests::intermediate_range_cases<DegreeRange>(Operation, basis.depth)) {
                if (ranges.multiplier.min > ranges.multiplier.max || ranges.multiplier.max > basis.depth ||
                    ranges.arg.min > ranges.arg.max || ranges.arg.max > basis.depth) {
                    continue;
                }
                SCOPED_TRACE(ranges.name);
                auto const multiplier = rpp::tests::make_intermediate_input(
                    Base::make_tensor(111, basis), basis, true, ranges.zero_multiplier);
                auto arg = rpp::tests::make_intermediate_input(
                    Base::make_tensor(112, basis), basis, false, ranges.zero_arg);
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
                    auto initial = Base::make_tensor(113, basis);
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
                    auto actual = initial;
                    rpp::DenseTensorView<Scalar*, Basis> out(actual.data() + 1, basis, 0, output_max);
                    rpp::DenseTensorView<Scalar const*, Basis> in(arg.data(), basis, ranges.arg.min, ranges.arg.max);
                    rpp::DenseTensorView<Scalar const*, Basis> mult(multiplier.data(), basis,
                                                                  ranges.multiplier.min, ranges.multiplier.max);
                    auto const ctx = Base::make_context();
                    if constexpr (Operation == Kind::Exp) {
                        rpp::ops::FTExp<Strategy>{}(ctx, out, in);
                    }
                    else if constexpr (Operation == Kind::Log) {
                        rpp::ops::FTLog<Strategy>{}(ctx, out, in);
                    }
                    else {
                        rpp::ops::FTFMExp<Strategy>{}(ctx, out, mult, in);
                    }
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
                    Base::expect_tensor_near(actual, expected);
                }
            }
        }
    }
};

TYPED_TEST_SUITE(CpuIntermediateInitializationTests, rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(CpuIntermediateInitializationTests, ExpInitializesOutputAndZeroExtendsOperands) {
    TestFixture::template check_output_initialization<rpp::tests::IntermediateOperation::Exp>();
}

TYPED_TEST(CpuIntermediateInitializationTests, LogInitializesOutputAndZeroExtendsOperands) {
    TestFixture::template check_output_initialization<rpp::tests::IntermediateOperation::Log>();
}

TYPED_TEST(CpuIntermediateInitializationTests, FMExpInitializesOutputAndZeroExtendsOperands) {
    TestFixture::template check_output_initialization<rpp::tests::IntermediateOperation::FMExp>();
}

} // namespace
