#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_adj_rmul.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_pairing.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class FreeTensorAdjointRightMulTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static Scalar one() { return make_scalar({{{}, 1, 1}}); }

    [[nodiscard]] static std::vector<Scalar> zero_tensor(Basis const& basis) {
        return std::vector<Scalar>(static_cast<std::size_t>(basis.size()));
    }

    [[nodiscard]] static Scalar pairing(Basis const& basis,
                                        std::vector<Scalar> const& lhs,
                                        std::vector<Scalar> const& rhs) {
        Scalar result;
        TensorView<Scalar const*> lhs_view(lhs.data(), basis);
        TensorView<Scalar const*> rhs_view(rhs.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorPairing<Strategy>{}(ctx, result, lhs_view, rhs_view);
        return result;
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

    [[nodiscard]] static std::vector<Scalar>
    apply_right_product(Basis const& basis,
                        std::vector<Scalar> const& arg,
                        std::vector<Scalar> const& op) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);
        TensorView<Scalar const*> op_view(op.data(), basis);

        auto const ctx = make_context();
        rpp::ops::FTMul<Strategy>{}(ctx, out_view, arg_view, op_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_adj_mul(Basis const& basis,
                  std::vector<Scalar> const& op,
                  std::vector<Scalar> const& arg) {
        using AdjMul = rpp::ops::FTAdjRMul<Strategy>;

        auto out = zero_tensor(basis);
        auto const scratch_bytes =
            AdjMul::scratch_space_size(Strategy{}, basis);
        std::vector<std::byte> scratch(scratch_bytes);

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> op_view(op.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = Strategy::make_context(scratch.data());
        AdjMul::init_scratch_space(ctx, basis);
        AdjMul{}(ctx, out_view, op_view, arg_view);
        AdjMul::destroy_scratch_space(ctx, basis);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_identity_operator(Basis const& basis) {
        auto result = zero_tensor(basis);
        result[0] = one();
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_letter_operator(Basis const& basis, Index letter_index) {
        auto result = zero_tensor(basis);
        result[static_cast<std::size_t>(basis.start_of_degree(1) + letter_index)] =
            one();
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    expected_right_shift(Basis const& basis,
                         std::vector<Scalar> const& arg,
                         Index letter_index) {
        auto result = zero_tensor(basis);
        for (Degree degree = 0; degree < basis.depth; ++degree) {
            const auto level_size = basis.size_of_degree(degree);
            const auto src_begin = basis.start_of_degree(degree + 1);
            const auto dst_begin = basis.start_of_degree(degree);
            for (Index idx = 0; idx < level_size; ++idx) {
                result[static_cast<std::size_t>(dst_begin + idx)] =
                    arg[static_cast<std::size_t>(
                        src_begin + idx * basis.width + letter_index)];
            }
        }
        return result;
    }
};

TEST_F(FreeTensorAdjointRightMulTests, SatisfiesAdjointPairingCriterion) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const op = make_tensor('a', basis);
    auto const t = make_tensor('t', basis);
    auto const arg = make_tensor('x', basis);

    auto const product = apply_right_product(basis, t, op);
    auto const adjoint = apply_adj_mul(basis, op, arg);

    EXPECT_EQ(pairing(basis, adjoint, t), pairing(basis, arg, product));
}

TEST_F(FreeTensorAdjointRightMulTests, IsBilinearInOperatorAndArgument) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const op1 = make_tensor('a', basis);
    auto const op2 = make_tensor('b', basis);
    auto const arg1 = make_tensor('x', basis);
    auto const arg2 = make_tensor('y', basis);

    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});
    auto const gamma = make_scalar({{{{'r', 3}}, 5, 1}});
    auto const delta = make_scalar({{{{'s', 4}}, 7, 1}});

    auto const op = linear_combo(op1, alpha, op2, beta);
    auto const arg = linear_combo(arg1, gamma, arg2, delta);

    auto const lhs = apply_adj_mul(basis, op, arg);

    auto const ax11 = apply_adj_mul(basis, op1, arg1);
    auto const ax12 = apply_adj_mul(basis, op1, arg2);
    auto const ax21 = apply_adj_mul(basis, op2, arg1);
    auto const ax22 = apply_adj_mul(basis, op2, arg2);

    std::vector<Scalar> rhs(static_cast<std::size_t>(basis.size()));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = (alpha * gamma) * ax11[i] + (alpha * delta) * ax12[i] +
            (beta * gamma) * ax21[i] + (beta * delta) * ax22[i];
    }

    EXPECT_EQ(lhs, rhs);
}

TEST_F(FreeTensorAdjointRightMulTests,
       IdentityOperatorReturnsArgumentForTruncatedView) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const op = make_identity_operator(basis);
    auto const arg = make_tensor('x', basis);

    using AdjMul = rpp::ops::FTAdjRMul<Strategy>;
    auto out = zero_tensor(basis);
    auto const scratch_bytes = AdjMul::scratch_space_size(Strategy{}, basis);
    std::vector<std::byte> scratch(scratch_bytes);

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> op_view(op.data(), basis, Degree{0}, Degree{0});
    TensorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = Strategy::make_context(scratch.data());
    AdjMul::init_scratch_space(ctx, basis);
    AdjMul{}(ctx, out_view, op_view, arg_view);
    AdjMul::destroy_scratch_space(ctx, basis);

    EXPECT_EQ(out, arg);
}

TEST_F(FreeTensorAdjointRightMulTests, LetterOperatorShiftsCoefficientsRight) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    constexpr Index letter_index = 1;
    auto const op = make_letter_operator(basis, letter_index);
    auto const arg = make_tensor('x', basis);

    using AdjMul = rpp::ops::FTAdjRMul<Strategy>;
    auto out = zero_tensor(basis);
    auto const scratch_bytes = AdjMul::scratch_space_size(Strategy{}, basis);
    std::vector<std::byte> scratch(scratch_bytes);

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> op_view(op.data(), basis, Degree{1}, Degree{1});
    TensorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = Strategy::make_context(scratch.data());
    AdjMul::init_scratch_space(ctx, basis);
    AdjMul{}(ctx, out_view, op_view, arg_view);
    AdjMul::destroy_scratch_space(ctx, basis);

    EXPECT_EQ(out, expected_right_shift(basis, arg, letter_index));
}

TEST_F(FreeTensorAdjointRightMulTests,
       KernelWrapperMatchesDirectOperationForIdentityOperator) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto op = Wrapper::make_batch('a', basis);
    auto const arg = Wrapper::make_batch('x', basis);

    for (Wrapper::Index tensor_idx = 0; tensor_idx < Wrapper::tensor_count;
         ++tensor_idx) {
        auto const offset =
            static_cast<std::size_t>(tensor_idx * basis.size());
        std::fill(op.begin() + offset, op.begin() + offset + basis.size(), Scalar{});
        op[offset] = one();
    }

    const auto err = rpp::ops::ft_adj_rmul(
        strategy,
        typename Strategy::LaunchConfig{},
        Wrapper::tensor_batch(actual, basis),
        rpp::make_tensor_batch(op.data(), basis.size(), Degree{0}, Degree{0}),
        Wrapper::tensor_batch(arg, basis),
        basis,
        Wrapper::tensor_count);

    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::FTAdjRMul<Wrapper::Strategy>>(
        basis, [&](auto const& op_impl, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto operator_arg =
                Wrapper::tensor_view(op, basis, tensor_idx).truncate(0, 0);
            auto operand = Wrapper::tensor_view(arg, basis, tensor_idx);
            op_impl(ctx, out, operator_arg, operand);
        });

    EXPECT_EQ(actual, expected);
}


} // namespace
