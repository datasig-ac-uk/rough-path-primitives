#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_inplace_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_mul.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

using Helper = rpp::tests::PolynomialTensorHelper;
using Degree = Helper::Degree;
using Scalar = Helper::Scalar;

struct DegreeRange {
    Degree min;
    Degree max;
};

struct InplaceMulViewCase {
    char const* name;
    DegreeRange lhs;
    DegreeRange rhs;
};

class FreeTensorInplaceMulTests : public testing::Test, public Helper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static TensorView<Scalar*> mutable_tensor_view(
        std::vector<Scalar>& data, Basis const& basis, DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static TensorView<Scalar const*>
    const_tensor_view(std::vector<Scalar> const& data,
                      Basis const& basis,
                      DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }

    static void
    expect_inplace_mul_matches_out_of_place(DegreeRange lhs_range,
                                            DegreeRange rhs_range,
                                            Scalar const& beta = Scalar{1}) {
        auto const basis_data = BasisData(width, depth);
        auto const& basis = basis_data.basis;

        auto const initial_lhs = make_tensor('a', basis);
        auto inplace_lhs = initial_lhs;
        auto expected = initial_lhs;
        auto const rhs = make_tensor('b', basis);

        auto inplace_lhs_view =
            mutable_tensor_view(inplace_lhs, basis, lhs_range);
        auto expected_view = mutable_tensor_view(expected, basis, lhs_range);
        auto const initial_lhs_view =
            const_tensor_view(initial_lhs, basis, lhs_range);
        auto const rhs_view = const_tensor_view(rhs, basis, rhs_range);

        auto const ctx = make_context();
        rpp::ops::FTInplaceMul<Strategy>{}(
            ctx, inplace_lhs_view, rhs_view, beta);
        rpp::ops::FTMul<Strategy>{}(
            ctx, expected_view, initial_lhs_view, rhs_view, beta);

        EXPECT_EQ(inplace_lhs, expected);
    }
};

TEST_F(FreeTensorInplaceMulTests, MatchesOutOfPlaceMulForFullViews) {
    expect_inplace_mul_matches_out_of_place({0, depth}, {0, depth});
}

TEST_F(FreeTensorInplaceMulTests, MatchesOutOfPlaceMulForTruncatedViews) {
    InplaceMulViewCase const cases[] = {
        {"lhs has positive min degree", {2, depth}, {0, depth}},
        {"lhs has truncated max degree", {0, 2}, {0, depth}},
        {"rhs has positive min degree", {0, depth}, {1, depth}},
        {"rhs has truncated max degree", {0, depth}, {0, 2}},
        {"both views are interior ranges", {1, 3}, {1, 2}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_inplace_mul_matches_out_of_place(test_case.lhs, test_case.rhs);
    }
}

TEST_F(FreeTensorInplaceMulTests, MatchesOutOfPlaceMulWithScaledProduct) {
    auto const beta = make_scalar({{{{'p', 1}}, 7, 3}});

    expect_inplace_mul_matches_out_of_place({1, depth}, {0, 2}, beta);
}

TEST_F(FreeTensorInplaceMulTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const beta = Wrapper::make_scalar({{{{'q', 2}}, 5, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const rhs = Wrapper::make_batch('b', basis);

    const auto err =
        rpp::ops::ft_inplace_mul(strategy,
                                 {},
                                 Wrapper::tensor_batch(actual, basis),
                                 Wrapper::tensor_batch(rhs, basis),
                                 basis,
                                 Wrapper::tensor_count,
                                 beta);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();

    Wrapper::apply_direct<rpp::ops::FTInplaceMul<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto lhs = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto right = Wrapper::tensor_view(rhs, basis, tensor_idx);
            op(ctx, lhs, right, beta);
        });

    EXPECT_EQ(actual, expected);
}

template <typename Config>
class NumericFreeTensorInplaceMulTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::ConstTensorView;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using typename Base::TensorView;
    using Base::const_tensor_view;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_tensor;
    using Base::make_unit_tensor;
    using Base::mutable_tensor_view;
    using Base::reference_mul;
    using Base::zero_tensor;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_inplace_mul(Basis const& basis,
                    std::vector<typename Base::Scalar> const& initial_lhs,
                    std::vector<typename Base::Scalar> const& rhs,
                    DegreeRange lhs_range,
                    DegreeRange rhs_range,
                    Accum beta = Accum{1}) {
        auto lhs = initial_lhs;

        auto lhs_view = mutable_tensor_view(lhs, basis, lhs_range);
        auto const rhs_view = const_tensor_view(rhs, basis, rhs_range);

        auto const ctx = Base::make_context();
        rpp::ops::FTInplaceMul<Strategy>{}(ctx, lhs_view, rhs_view, beta);
        return lhs;
    }
};

TYPED_TEST_SUITE(NumericFreeTensorInplaceMulTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericFreeTensorInplaceMulTests, MatchesOutOfPlaceReferenceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const beta = typename TestFixture::Accum{1.5};

    auto const initial_lhs = TestFixture::make_tensor(1, basis);
    auto const rhs = TestFixture::make_tensor(2, basis);

    auto const actual = TestFixture::run_inplace_mul(
        basis,
        initial_lhs,
        rhs,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        beta);
    auto const expected = TestFixture::reference_mul(
        basis,
        initial_lhs,
        initial_lhs,
        rhs,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        beta);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericFreeTensorInplaceMulTests,
           MatchesOutOfPlaceReferenceForTruncatedViews) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    typename TestFixture::DegreeRange const lhs_range{1, 3};
    typename TestFixture::DegreeRange const rhs_range{1, 2};

    auto const initial_lhs = TestFixture::make_tensor(3, basis);
    auto const rhs = TestFixture::make_tensor(4, basis);

    auto const actual = TestFixture::run_inplace_mul(
        basis, initial_lhs, rhs, lhs_range, rhs_range);
    auto const expected = TestFixture::reference_mul(
        basis, initial_lhs, initial_lhs, rhs, lhs_range, lhs_range, rhs_range);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericFreeTensorInplaceMulTests, RightUnitMatchesOutOfPlaceReference) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const beta = typename TestFixture::Accum{0.75};

    auto const initial_lhs = TestFixture::make_tensor(5, basis);
    auto const unit = TestFixture::make_unit_tensor(basis);

    auto const actual = TestFixture::run_inplace_mul(
        basis,
        initial_lhs,
        unit,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        beta);
    auto const expected = TestFixture::reference_mul(
        basis,
        initial_lhs,
        initial_lhs,
        unit,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        beta);
    TestFixture::expect_tensor_near(actual, expected);
}

} // namespace
