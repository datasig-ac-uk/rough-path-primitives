#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_fma.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_inplace_fma.hpp>
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

struct InplaceFmaViewCase {
    char const* name;
    DegreeRange a;
    DegreeRange b;
    DegreeRange c;
};

class FreeTensorInplaceFmaTests
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

    static void expect_inplace_fma_matches_out_of_place(
        DegreeRange a_range,
        DegreeRange b_range,
        DegreeRange c_range,
        Scalar const& alpha = Scalar{1},
        Scalar const& beta = Scalar{1}
    )
    {
        auto const basis_data = BasisData(width, depth);
        auto const& basis = basis_data.basis;

        auto const initial_a = make_tensor('a', basis);
        auto inplace_a = initial_a;
        auto expected = initial_a;
        auto const b = make_tensor('b', basis);
        auto const c = make_tensor('c', basis);

        auto inplace_a_view = mutable_tensor_view(inplace_a, basis, a_range);
        auto expected_view = mutable_tensor_view(expected, basis, a_range);
        auto const initial_a_view = const_tensor_view(initial_a, basis, a_range);
        auto const b_view = const_tensor_view(b, basis, b_range);
        auto const c_view = const_tensor_view(c, basis, c_range);

        auto const ctx = make_context();
        rpp::ops::FTInplaceFma231<Strategy>{}(
            ctx,
            inplace_a_view,
            b_view,
            c_view,
            alpha,
            beta
        );
        rpp::ops::FTFma<Strategy>{}(
            ctx,
            expected_view,
            initial_a_view,
            b_view,
            c_view,
            alpha,
            beta
        );

        EXPECT_EQ(inplace_a, expected);
    }
};

TEST_F(FreeTensorInplaceFmaTests, MatchesOutOfPlaceFmaForFullViews)
{
    expect_inplace_fma_matches_out_of_place(
        {0, depth},
        {0, depth},
        {0, depth}
    );
}

TEST_F(FreeTensorInplaceFmaTests, MatchesOutOfPlaceFmaForTruncatedInputs)
{
    InplaceFmaViewCase const cases[] = {
        {"a has positive min degree", {2, depth}, {0, depth}, {0, depth}},
        {"a has truncated max degree", {0, 2}, {0, depth}, {0, depth}},
        {"b has positive min degree", {0, depth}, {1, depth}, {0, depth}},
        {"c has positive min degree", {0, depth}, {0, depth}, {1, depth}},
        {"b and c have truncated max degrees", {0, depth}, {0, 2}, {0, 1}},
        {"all views are interior ranges", {1, 3}, {1, 2}, {1, 3}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_inplace_fma_matches_out_of_place(
            test_case.a,
            test_case.b,
            test_case.c
        );
    }
}

TEST_F(FreeTensorInplaceFmaTests, MatchesOutOfPlaceFmaWithScaledAddendAndProduct)
{
    auto const alpha = make_scalar({{{{'p', 1}}, 3, 2}});
    auto const beta = make_scalar({{{{'q', 2}}, 5, 3}});

    expect_inplace_fma_matches_out_of_place(
        {1, depth},
        {0, 2},
        {1, depth},
        alpha,
        beta
    );
}

TEST_F(FreeTensorInplaceFmaTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const alpha = Wrapper::make_scalar({{{{'p', 1}}, 3, 2}});
    auto const beta = Wrapper::make_scalar({{{{'q', 2}}, 5, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const b = Wrapper::make_batch('b', basis);
    auto const c = Wrapper::make_batch('c', basis);

    const auto err = rpp::ops::ft_inplace_fma<rpp::ops::FTInplaceFMAType::AEqualsBCPlusA>(
        strategy,
        {},
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(b, basis),
        Wrapper::tensor_batch(c, basis),
        basis,
        Wrapper::tensor_count,
        alpha,
        beta
        );
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();

    Wrapper::apply_direct<rpp::ops::FTInplaceFma231<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto a = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto left = Wrapper::tensor_view(b, basis, tensor_idx);
            auto right = Wrapper::tensor_view(c, basis, tensor_idx);
            op(ctx, a, left, right, alpha, beta);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
