#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/ft_inplace_mul.hpp>
#include <rpp/cpu/operations/single_thread/basic/ft_mul.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
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

class FreeTensorInplaceMulTests
    : public testing::Test,
      public Helper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static TensorView<Scalar*> mutable_tensor_view(
        std::vector<Scalar>& data,
        Basis const& basis,
        DegreeRange range
    )
    {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static TensorView<Scalar const*> const_tensor_view(
        std::vector<Scalar> const& data,
        Basis const& basis,
        DegreeRange range
    )
    {
        return {data.data(), basis, range.min, range.max};
    }

    static void expect_inplace_mul_matches_out_of_place(
        DegreeRange lhs_range,
        DegreeRange rhs_range,
        Scalar const& beta = Scalar{1}
    )
    {
        auto const basis_data = BasisData(width, depth);
        auto const& basis = basis_data.basis;

        auto const initial_lhs = make_tensor('a', basis);
        auto inplace_lhs = initial_lhs;
        auto expected = initial_lhs;
        auto const rhs = make_tensor('b', basis);

        auto inplace_lhs_view = mutable_tensor_view(inplace_lhs, basis, lhs_range);
        auto expected_view = mutable_tensor_view(expected, basis, lhs_range);
        auto const initial_lhs_view = const_tensor_view(initial_lhs, basis, lhs_range);
        auto const rhs_view = const_tensor_view(rhs, basis, rhs_range);

        auto const ctx = make_context();
        rpp::ops::FTInplaceMul<Strategy>{}(
            ctx,
            inplace_lhs_view,
            rhs_view,
            beta
        );
        rpp::ops::FTMul<Strategy>{}(
            ctx,
            expected_view,
            initial_lhs_view,
            rhs_view,
            beta
        );

        EXPECT_EQ(inplace_lhs, expected);
    }
};

TEST_F(FreeTensorInplaceMulTests, MatchesOutOfPlaceMulForFullViews)
{
    expect_inplace_mul_matches_out_of_place(
        {0, depth},
        {0, depth}
    );
}

TEST_F(FreeTensorInplaceMulTests, MatchesOutOfPlaceMulForTruncatedViews)
{
    InplaceMulViewCase const cases[] = {
        {"lhs has positive min degree", {2, depth}, {0, depth}},
        {"lhs has truncated max degree", {0, 2}, {0, depth}},
        {"rhs has positive min degree", {0, depth}, {1, depth}},
        {"rhs has truncated max degree", {0, depth}, {0, 2}},
        {"both views are interior ranges", {1, 3}, {1, 2}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_inplace_mul_matches_out_of_place(
            test_case.lhs,
            test_case.rhs
        );
    }
}

TEST_F(FreeTensorInplaceMulTests, MatchesOutOfPlaceMulWithScaledProduct)
{
    auto const beta = make_scalar({{{{'p', 1}}, 7, 3}});

    expect_inplace_mul_matches_out_of_place(
        {1, depth},
        {0, 2},
        beta
    );
}

TEST_F(FreeTensorInplaceMulTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const beta = Wrapper::make_scalar({{{{'q', 2}}, 5, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const rhs = Wrapper::make_batch('b', basis);

    const auto err = rpp::ops::ft_inplace_mul(
        strategy,
        {},
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(rhs, basis),
        basis,
        Wrapper::tensor_count,
        beta
        );
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();

    Wrapper::apply_direct<rpp::ops::FTInplaceMul<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto lhs = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto right = Wrapper::tensor_view(rhs, basis, tensor_idx);
            op(ctx, lhs, right, beta);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
