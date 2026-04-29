#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/ops/single_thread/tensor_reflect.hpp>
#include <rpp/dense/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class TensorReflectTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static std::vector<Scalar> apply_reflect(
        Basis const& basis,
        std::vector<Scalar> const& arg
    )
    {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorReflect<Strategy>{}(ctx, out_view, arg_view);
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

TEST_F(TensorReflectTests, AppliesUnsignedWordReversal)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);
    auto const out = apply_reflect(basis, arg);

    std::vector<Scalar> expected(static_cast<std::size_t>(basis.size()));
    for_each_index(
        basis,
        [&expected, &arg, &basis](Degree degree, Index level_index) {
            auto const reversed = reverse_index(basis, degree, level_index);
            auto const source = basis.start_of_degree(degree) + level_index;
            auto const target = basis.start_of_degree(degree) + reversed;

            expected[static_cast<std::size_t>(target)] =
                arg[static_cast<std::size_t>(source)];
        }
    );

    EXPECT_EQ(out, expected);
}

TEST_F(TensorReflectTests, IsLinear)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);
    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});

    auto const arg = linear_combo(lhs, alpha, rhs, beta);
    auto const reflected_arg = apply_reflect(basis, arg);
    auto const reflected_lhs = apply_reflect(basis, lhs);
    auto const reflected_rhs = apply_reflect(basis, rhs);

    EXPECT_EQ(
        reflected_arg,
        linear_combo(reflected_lhs, alpha, reflected_rhs, beta)
    );
}

TEST_F(TensorReflectTests, IsInvolution)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const arg = make_tensor('a', basis);

    EXPECT_EQ(apply_reflect(basis, apply_reflect(basis, arg)), arg);
}

TEST_F(TensorReflectTests, KernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('a', basis);

    rpp::cpu::single_thread::tensor_reflect_kernel(
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(arg, basis),
        basis,
        strategy,
        Wrapper::tensor_count
    );
    Wrapper::apply_direct<rpp::ops::TensorReflect<Wrapper::Strategy>>(
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
