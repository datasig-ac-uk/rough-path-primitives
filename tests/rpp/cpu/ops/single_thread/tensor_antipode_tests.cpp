#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/operations/single_thread/basic/tensor_antipode.hpp>
#include <rpp/dense/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class TensorAntipodeTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static std::vector<Scalar> apply_antipode(
        Basis const& basis,
        std::vector<Scalar> const& arg
    )
    {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorAntipode<Strategy>{}(ctx, out_view, arg_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar> linear_combo(
        std::vector<Scalar> const& lhs,
        Scalar const& lhs_scale,
        std::vector<Scalar> const& rhs,
        Scalar const& rhs_scale
    )
    {
        std::vector<Scalar> result(lhs.size());
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            result[i] = lhs_scale * lhs[i] + rhs_scale * rhs[i];
        }
        return result;
    }
};

TEST_F(TensorAntipodeTests, AppliesSignedWordReversal)
{
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
        basis,
        [&expected, &arg, &basis](Degree degree, Index level_index) {
            auto const reversed = reverse_index(basis, degree, level_index);
            auto const source = basis.start_of_degree(degree) + level_index;
            auto const target = basis.start_of_degree(degree) + reversed;

            expected[static_cast<std::size_t>(target)] =
                degree % 2 == 0
                    ? arg[static_cast<std::size_t>(source)]
                    : -arg[static_cast<std::size_t>(source)];
        }
    );

    EXPECT_EQ(out, expected);
}

TEST_F(TensorAntipodeTests, IsLinear)
{
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

    EXPECT_EQ(
        antipode_arg,
        linear_combo(antipode_lhs, alpha, antipode_rhs, beta)
    );
}

TEST_F(TensorAntipodeTests, IsInvolution)
{
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

TEST_F(TensorAntipodeTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('a', basis);

    rpp::cpu::single_thread::tensor_antipode_kernel(
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(arg, basis),
        basis,
        strategy,
        Wrapper::tensor_count
    );
    Wrapper::apply_direct<rpp::ops::TensorAntipode<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto in = Wrapper::tensor_view(arg, basis, tensor_idx);
            op(ctx, out, in);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
