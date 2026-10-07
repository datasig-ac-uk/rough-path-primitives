#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/left_hs_adj_lmul.hpp>
#include <rpp/cpu/single_thread/operations/basic/left_hs_adj_rmul.hpp>
#include <rpp/cpu/single_thread/operations/basic/left_hs_mul.hpp>

#include "../../../half_shuffle_adjoint_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"

namespace {

template <typename Config>
class CpuHalfShuffleAdjointTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::ConstTensorView;
    using typename Base::Degree;
    using typename Base::DegreeRange;
    using typename Base::Index;
    using typename Base::Scalar;
    using typename Base::Strategy;
    using typename Base::TensorView;

    template <bool FixedLeft>
    static std::vector<Scalar> run_adjoint(Basis const& basis,
                                           std::vector<Scalar> const& initial,
                                           std::vector<Scalar> const& op,
                                           std::vector<Scalar> const& arg,
                                           DegreeRange out_range,
                                           DegreeRange op_range,
                                           DegreeRange arg_range) {
        auto actual = initial;
        TensorView out(actual.data() + 1, basis, out_range.min, out_range.max);
        ConstTensorView fixed_operand(
            op.data(), basis, op_range.min, op_range.max);
        ConstTensorView cotangent(
            arg.data(), basis, arg_range.min, arg_range.max);
        if constexpr (FixedLeft) {
            rpp::ops::LeftHSAdjLMul<Strategy>{}(
                Base::make_context(), out, fixed_operand, cotangent);
        }
        else {
            // The right-adjoint interface takes cotangent, fixed integrand.
            rpp::ops::LeftHSAdjRMul<Strategy>{}(
                Base::make_context(), out, cotangent, fixed_operand);
        }
        return actual;
    }

    static std::vector<Scalar> run_forward(Basis const& basis,
                                           std::vector<Scalar> const& initial,
                                           std::vector<Scalar> const& op,
                                           std::vector<Scalar> const& arg,
                                           DegreeRange out_range,
                                           DegreeRange op_range,
                                           DegreeRange arg_range,
                                           Accum beta) {
        auto actual = initial;
        TensorView out(actual.data() + 1, basis, out_range.min, out_range.max);
        ConstTensorView op_view(op.data(), basis, op_range.min, op_range.max);
        ConstTensorView arg_view(
            arg.data(), basis, arg_range.min, arg_range.max);
        auto const ctx = Base::make_context();
        rpp::ops::LeftHSMul<Strategy>{}(ctx, out, op_view, arg_view, beta);
        return actual;
    }

    template <bool FixedLeft>
    static void check_ranges() {
        struct BasisConfig {
            Degree width, depth;
        };
        BasisConfig const configs[] = {{1, 4}, {2, 2}, {4, 4}, {4, 0}};
        for (auto const& config : configs) {
            SCOPED_TRACE(testing::Message() << "width=" << config.width
                                            << ", depth=" << config.depth);
            auto const basis_data =
                typename Base::BasisData(config.width, config.depth);
            auto const& basis = basis_data.basis;
            for (auto const& ranges :
                 rpp::tests::half_shuffle_adjoint_range_cases<DegreeRange>(
                     basis.depth)) {
                SCOPED_TRACE(testing::Message()
                             << ranges.name << " out=[" << ranges.out.min << ","
                             << ranges.out.max << "]"
                             << " op=[" << ranges.op.min << "," << ranges.op.max
                             << "]"
                             << " arg=[" << ranges.arg.min << ","
                             << ranges.arg.max << "]");
                for (float beta : {1.0f}) {
                    SCOPED_TRACE(beta);
                    auto op =
                        rpp::tests::make_half_shuffle_adjoint_operator<Scalar>(
                            basis);
                    auto arg =
                        rpp::tests::make_half_shuffle_cotangent<Scalar>(basis);
                    auto const coefficients =
                        rpp::tests::reference_half_shuffle_adjoint(basis,
                                                                   op,
                                                                   arg,
                                                                   ranges.out,
                                                                   ranges.op,
                                                                   ranges.arg,
                                                                   FixedLeft,
                                                                   beta);
                    // Populate inactive input storage with NaNs to catch
                    // forbidden reads.
                    for (Degree degree = 0; degree <= basis.depth; ++degree) {
                        for (auto idx = basis.start_of_degree(degree);
                             idx < basis.end_of_degree(degree);
                             ++idx) {
                            if (degree < ranges.op.min ||
                                degree > ranges.op.max) {
                                op[idx] = static_cast<Scalar>(
                                    std::numeric_limits<float>::quiet_NaN());
                            }
                            if (degree < ranges.arg.min ||
                                degree > ranges.arg.max) {
                                arg[idx] = static_cast<Scalar>(
                                    std::numeric_limits<float>::quiet_NaN());
                            }
                        }
                    }
                    std::vector<Scalar> initial(
                        static_cast<std::size_t>(basis.size()) + 2,
                        static_cast<Scalar>(7.0f));
                    initial.back() = static_cast<Scalar>(-7.0f);
                    auto expected = initial;
                    auto const begin = basis.start_of_degree(ranges.out.min);
                    auto const end = basis.end_of_degree(ranges.out.max);
                    for (auto idx = begin; idx < end; ++idx) {
                        initial[idx + 1] = static_cast<Scalar>(
                            std::numeric_limits<float>::quiet_NaN());
                        expected[idx + 1] =
                            static_cast<Scalar>(coefficients[idx]);
                    }
                    auto const actual = run_adjoint<FixedLeft>(basis,
                                                               initial,
                                                               op,
                                                               arg,
                                                               ranges.out,
                                                               ranges.op,
                                                               ranges.arg);
                    Base::expect_tensor_near(actual, expected);
                    for (std::size_t i = 0; i < actual.size(); ++i) {
                        if (i <= static_cast<std::size_t>(begin) ||
                            i > static_cast<std::size_t>(end)) {
                            EXPECT_EQ(static_cast<double>(actual[i]),
                                      static_cast<double>(expected[i]))
                                << i;
                        }
                        else {
                            EXPECT_TRUE(
                                std::isfinite(static_cast<double>(actual[i])))
                                << i;
                            if (coefficients[i - 1] == 0.0) {
                                EXPECT_EQ(static_cast<double>(actual[i]), 0.0)
                                    << i;
                            }
                        }
                    }
                }
            }
        }
    }

    template <bool FixedLeft>
    static void check_pairing() {
        auto const basis_data = typename Base::BasisData(3, 3);
        auto const& basis = basis_data.basis;
        auto const integrator =
            rpp::tests::make_half_shuffle_adjoint_operator<Scalar>(basis);
        auto const integrand =
            rpp::tests::make_half_shuffle_cotangent<Scalar>(basis);
        auto cotangent = integrand;
        for (std::size_t i = 0; i < cotangent.size(); ++i) {
            cotangent[i] =
                static_cast<Scalar>(float(int((i * 11) % 7) - 3) / 8.0f);
        }
        struct Ranges {
            DegreeRange integrator, integrand, cotangent;
        };
        Ranges const cases[] = {
            {{0, 3}, {0, 3}, {0, 3}},
            {{1, 2}, {0, 1}, {2, 3}},
            {{0, 0}, {0, 3}, {0, 3}},
            {{1, 1}, {0, 0}, {1, 1}},
            {{0, 3}, {0, 0}, {1, 2}},
        };
        std::vector<Scalar> initial(static_cast<std::size_t>(basis.size()) + 2,
                                    std::numeric_limits<Scalar>::quiet_NaN());
        for (auto const& ranges : cases) {
            auto const out_range =
                FixedLeft ? ranges.integrand : ranges.integrator;
            auto const op_range =
                FixedLeft ? ranges.integrator : ranges.integrand;
            auto const& fixed_operand = FixedLeft ? integrator : integrand;
            auto const& varying_operand = FixedLeft ? integrand : integrator;
            auto const adjoint = run_adjoint<FixedLeft>(basis,
                                                        initial,
                                                        fixed_operand,
                                                        cotangent,
                                                        out_range,
                                                        op_range,
                                                        ranges.cotangent);
            auto const product = run_forward(basis,
                                             initial,
                                             integrator,
                                             integrand,
                                             ranges.cotangent,
                                             ranges.integrator,
                                             ranges.integrand,
                                             Accum{1});
            double lhs = 0, rhs = 0;
            for (auto i = basis.start_of_degree(ranges.cotangent.min);
                 i < basis.end_of_degree(ranges.cotangent.max);
                 ++i) {
                lhs += double(cotangent[i]) * double(product[i + 1]);
            }
            for (auto i = basis.start_of_degree(out_range.min);
                 i < basis.end_of_degree(out_range.max);
                 ++i) {
                rhs += double(varying_operand[i]) * double(adjoint[i + 1]);
            }
            EXPECT_NEAR(lhs, rhs, 1e-10);
        }
    }

    template <bool FixedLeft>
    static void check_zero_inputs() {
        auto const basis_data = typename Base::BasisData(Degree{4}, Degree{4});
        auto const& basis = basis_data.basis;
        DegreeRange const range{0, basis.depth};
        for (bool zero_operator : {false, true}) {
            auto op =
                rpp::tests::make_half_shuffle_adjoint_operator<Scalar>(basis);
            auto arg = rpp::tests::make_half_shuffle_cotangent<Scalar>(basis);
            auto& zero_input = zero_operator ? op : arg;
            std::fill(zero_input.begin(),
                      zero_input.end(),
                      static_cast<Scalar>(0.0f));
            std::vector<Scalar> initial(
                static_cast<std::size_t>(basis.size()) + 2,
                static_cast<Scalar>(std::numeric_limits<float>::quiet_NaN()));
            initial.front() = static_cast<Scalar>(7.0f);
            initial.back() = static_cast<Scalar>(-7.0f);
            auto const actual = run_adjoint<FixedLeft>(
                basis, initial, op, arg, range, range, range);
            EXPECT_EQ(static_cast<double>(actual.front()), 7.0);
            EXPECT_EQ(static_cast<double>(actual.back()), -7.0);
            for (std::size_t i = 1; i + 1 < actual.size(); ++i) {
                EXPECT_EQ(static_cast<double>(actual[i]), 0.0) << i;
            }
        }
    }
};

TYPED_TEST_SUITE(CpuHalfShuffleAdjointTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(CpuHalfShuffleAdjointTests,
           FixedLeftAdjointMatchesTransposeAndRespectsBounds) {
    TestFixture::template check_ranges<true>();
}

TYPED_TEST(CpuHalfShuffleAdjointTests,
           ZeroInputsOverwriteEveryOutputCoefficient) {
    TestFixture::template check_zero_inputs<true>();
}

TYPED_TEST(CpuHalfShuffleAdjointTests, SatisfiesFixedLeftAdjointPairing) {
    TestFixture::template check_pairing<true>();
}

TYPED_TEST(CpuHalfShuffleAdjointTests,
           FixedRightAdjointMatchesTransposeAndRespectsBounds) {
    TestFixture::template check_ranges<false>();
}

TYPED_TEST(CpuHalfShuffleAdjointTests,
           FixedRightZeroInputsOverwriteEveryOutputCoefficient) {
    TestFixture::template check_zero_inputs<false>();
}

TYPED_TEST(CpuHalfShuffleAdjointTests, SatisfiesFixedRightAdjointPairing) {
    TestFixture::template check_pairing<false>();
}

} // namespace
