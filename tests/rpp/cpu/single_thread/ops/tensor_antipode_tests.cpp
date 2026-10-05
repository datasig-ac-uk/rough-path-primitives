#include <vector>

#include <gtest/gtest.h>

#include "../../../tensor_antipode_test_helper.hpp"

#include <rpp/cpu/single_thread/operations/basic/ft_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_antipode.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

template <typename Config>
class NumericTensorAntipodeTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::ConstTensorView;
    using typename Base::Degree;
    using typename Base::Strategy;
    using typename Base::TensorView;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::linear_combo;
    using Base::make_tensor;
    using Base::mutable_tensor_view;
    using Base::reference_mul;
    using Base::zero_tensor;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    make_group_like_argument(Basis const& basis, unsigned seed) {
        auto group_like = zero_tensor(basis);
        auto const lambda =
            static_cast<Accum>((static_cast<unsigned>(seed % 3u) + 1u) *
                               0.03125);

        group_like[0] = static_cast<typename Base::Scalar>(Accum{1});
        Accum coeff{1};
        for (Degree degree = 1; degree <= basis.depth; ++degree) {
            coeff = coeff * lambda / static_cast<Accum>(degree);
            group_like[static_cast<std::size_t>(basis.start_of_degree(degree))] =
                static_cast<typename Base::Scalar>(coeff);
        }
        return group_like;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_antipode(Basis const& basis,
                 std::vector<typename Base::Scalar> const& arg) {
        auto out = zero_tensor(basis);
        TensorView out_view(out.data(), basis);
        ConstTensorView arg_view(arg.data(), basis);

        auto const ctx = Base::make_context();
        rpp::ops::TensorAntipode<Strategy>{}(ctx, out_view, arg_view);
        return out;
    }
};

TYPED_TEST_SUITE(NumericTensorAntipodeTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericTensorAntipodeTests, IsLinear) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const alpha = typename TestFixture::Accum{0.5};
    auto const beta = typename TestFixture::Accum{-1.25};

    auto const lhs = TestFixture::make_tensor(1, basis);
    auto const rhs = TestFixture::make_tensor(2, basis);
    auto const arg = TestFixture::linear_combo(lhs, alpha, rhs, beta);

    auto const antipode_arg = TestFixture::run_antipode(basis, arg);
    auto const antipode_lhs = TestFixture::run_antipode(basis, lhs);
    auto const antipode_rhs = TestFixture::run_antipode(basis, rhs);
    auto const expected =
        TestFixture::linear_combo(antipode_lhs, alpha, antipode_rhs, beta);

    TestFixture::expect_tensor_near(antipode_arg, expected);
}

TYPED_TEST(NumericTensorAntipodeTests, ActsAsInverseOnGroupLikeTensor) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    auto const arg = TestFixture::make_group_like_argument(basis, 7);
    auto const antipode = TestFixture::run_antipode(basis, arg);
    auto const zero = TestFixture::zero_tensor(basis);
    auto const product = TestFixture::reference_mul(
        basis,
        zero,
        antipode,
        arg,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis));
    auto const expected = TestFixture::make_unit_tensor(basis);

    TestFixture::expect_tensor_near(product, expected);
}

class TensorAntipodeTests : public testing::Test,
                            public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static std::vector<Scalar>
    apply_antipode(Basis const& basis, std::vector<Scalar> const& arg) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorAntipode<Strategy>{}(ctx, out_view, arg_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    linear_combo(std::vector<Scalar> const& lhs,
                 Scalar const& lhs_scale,
                 std::vector<Scalar> const& rhs,
                 Scalar const& rhs_scale) {
        std::vector<Scalar> result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = lhs_scale * lhs[i] + rhs_scale * rhs[i];
        }
        return result;
    }
};

TEST_F(TensorAntipodeTests, AppliesSignedWordReversal) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);
    std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));
    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = make_context();
    rpp::ops::TensorAntipode<Strategy>{}(ctx, out_view, arg_view);

    for_each_index(
        basis, [&expected, &arg, &basis](Degree degree, Index level_index) {
            auto const reversed = reverse_index(basis, degree, level_index);
            auto const source = basis.start_of_degree(degree) + level_index;
            auto const target = basis.start_of_degree(degree) + reversed;

            expected[static_cast<std::size_t>(target)] = degree % 2 == 0
                ? arg[static_cast<std::size_t>(source)]
                : -arg[static_cast<std::size_t>(source)];
        });

    EXPECT_EQ(out, expected);
}

TEST_F(TensorAntipodeTests, IsLinear) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);
    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});

    auto const arg = linear_combo(lhs, alpha, rhs, beta);
    auto const antipode_arg = apply_antipode(basis, arg);
    auto const antipode_lhs = apply_antipode(basis, lhs);
    auto const antipode_rhs = apply_antipode(basis, rhs);

    EXPECT_EQ(antipode_arg,
              linear_combo(antipode_lhs, alpha, antipode_rhs, beta));
}

TEST_F(TensorAntipodeTests, IsInvolution) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);
    std::vector<Scalar> tmp(static_cast<std::size_t>(basis.size()));
    std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

    TensorView<Scalar*> tmp_view(tmp.data(), basis);
    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> arg_view(arg.data(), basis);
    TensorView<Scalar const*> tmp_const_view(tmp.data(), basis);

    auto const ctx = make_context();
    rpp::ops::TensorAntipode<Strategy>{}(ctx, tmp_view, arg_view);
    rpp::ops::TensorAntipode<Strategy>{}(ctx, out_view, tmp_const_view);

    EXPECT_EQ(out, arg);
}

TEST_F(TensorAntipodeTests, AppliesOnlyToOverlappingArgumentDegreeRange) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);
    std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));
    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> arg_view(arg.data(), basis, 2, 3);

    auto const ctx = make_context();
    rpp::ops::TensorAntipode<Strategy>{}(ctx, out_view, arg_view);

    for_each_index(
        basis, [&expected, &arg, &basis](Degree degree, Index level_index) {
            if (degree < 2 || degree > 3) {
                return;
            }
            auto const reversed = reverse_index(basis, degree, level_index);
            auto const source = basis.start_of_degree(degree) + level_index;
            auto const target = basis.start_of_degree(degree) + reversed;

            expected[static_cast<std::size_t>(target)] = degree % 2 == 0
                ? arg[static_cast<std::size_t>(source)]
                : -arg[static_cast<std::size_t>(source)];
        });

    EXPECT_EQ(out, expected);
}

TEST_F(TensorAntipodeTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('a', basis);

    auto const err =
        rpp::ops::tensor_antipode(strategy,
                                  typename Wrapper::Strategy::LaunchConfig{},
                                  Wrapper::tensor_batch(actual, basis),
                                  Wrapper::tensor_batch(arg, basis),
                                  basis,
                                  Wrapper::tensor_count);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::TensorAntipode<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto in = Wrapper::tensor_view(arg, basis, tensor_idx);
            op(ctx, out, in);
        });

    EXPECT_EQ(actual, expected);
}

TYPED_TEST(NumericTensorAntipodeTests, ZeroExtendsArgumentAndPreservesOutsideOutputView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    using Scalar = typename TestFixture::Scalar;
    using Degree = typename TestFixture::Degree;
    using Index = typename TestFixture::Index;
    using DegreeRange = typename TestFixture::DegreeRange;
    auto const& basis = basis_data.basis;
    std::vector<Scalar> arg(static_cast<std::size_t>(basis.size()));
    for (std::size_t i = 0; i < arg.size(); ++i) {
        arg[i] = static_cast<Scalar>(i + 1);
    }
    std::vector<Scalar> initial_out(arg.size(), Scalar{7});

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
                    expected[target] = Scalar{0};
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

        typename TestFixture::TensorView out_view(
            actual.data(), basis, test_case.out.min, test_case.out.max);
        typename TestFixture::ConstTensorView arg_view(
            arg.data(), basis, test_case.arg.min, test_case.arg.max);
        rpp::ops::TensorAntipode<typename TestFixture::Strategy>{}(
            TestFixture::make_context(), out_view, arg_view);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

TYPED_TEST(NumericTensorAntipodeTests, HandlesEveryInputAndOutputDegreeRange) {
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
        auto const basis_data = typename TestFixture::BasisData(config.width, config.depth);
        auto const& basis = basis_data.basis;
        auto initial_out = TestFixture::make_tensor(71, basis);
        for (auto& value : initial_out) {
            value = static_cast<Scalar>(-7.0f);
        }
        auto const arg = rpp::tests::make_antipode_range_argument(
            TestFixture::make_tensor(72, basis));
        auto const ranges = rpp::tests::all_tensor_degree_ranges<Range>(basis.depth);
        for (auto const out_range : ranges) {
            for (auto const arg_range : ranges) {
                SCOPED_TRACE(testing::Message()
                             << "out=[" << out_range.min << ',' << out_range.max
                             << "], arg=[" << arg_range.min << ',' << arg_range.max << ']');
                auto const expected = rpp::tests::reference_tensor_antipode<
                    typename TestFixture::Accum>(
                    initial_out, arg, basis, out_range, arg_range,
                    true);
                auto actual = initial_out;
                typename TestFixture::TensorView out_view(
                    actual.data(), basis, out_range.min, out_range.max);
                typename TestFixture::ConstTensorView arg_view(
                    arg.data(), basis, arg_range.min, arg_range.max);
                rpp::ops::TensorGeneralisedAntipode<typename TestFixture::Strategy,
                    rpp::ops::TensorAntipodeSigningPolicy::SignByDegree>{}(
                    TestFixture::make_context(), out_view, arg_view);
                EXPECT_EQ(actual, expected);
            }
        }
    }
}

} // namespace
