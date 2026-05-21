#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_inplace_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_set_identity.hpp>
#include <rpp/cpu/single_thread/operations/intermediate/ft_exp.hpp>
#include <rpp/cpu/single_thread/operations/intermediate/ft_fmexp.hpp>
#include <rpp/cpu/single_thread/operations/linalg/vector_assign.hpp>
#include <rpp/cpu/single_thread/operations/linalg/vector_inplace_add.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class FreeTensorFMExpTests : public testing::Test,
                             public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 2;
    static constexpr Degree depth = 4;

    [[nodiscard]] static std::vector<Scalar>
    make_positive_degree_tensor(char marker, Basis const& basis) {
        auto result = make_tensor(marker, basis);
        result[0] = Scalar{0};
        return result;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_exp(Basis const& basis, std::vector<Scalar> const& arg) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::FTExp<Strategy>{}(ctx, out_view, arg_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_mul(Basis const& basis,
              std::vector<Scalar> const& lhs,
              std::vector<Scalar> const& rhs) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> lhs_view(lhs.data(), basis);
        TensorView<Scalar const*> rhs_view(rhs.data(), basis);

        auto const ctx = make_context();
        rpp::ops::FTMul<Strategy>{}(ctx, out_view, lhs_view, rhs_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_fmexp(Basis const& basis,
                std::vector<Scalar> const& multiplier,
                std::vector<Scalar> const& exponent) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> multiplier_view(multiplier.data(), basis);
        TensorView<Scalar const*> exponent_view(exponent.data(), basis);

        auto const ctx = make_context();
        rpp::ops::FTFMExp<Strategy>{}(
            ctx, out_view, multiplier_view, exponent_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_fmexp_untruncated_horner(Basis const& basis,
                                   std::vector<Scalar> const& multiplier,
                                   std::vector<Scalar> const& exponent) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> multiplier_view(multiplier.data(), basis);
        TensorView<Scalar const*> exponent_view(exponent.data(), basis);

        auto const ctx = make_context();
        rpp::ops::VectorAssign<Strategy>{}(ctx, out_view, multiplier_view);

        Scalar const one{1};
        for (Degree d = basis.depth; d > 0; --d) {
            rpp::ops::FTInplaceMul<Strategy>{}(
                ctx, out_view, exponent_view.truncate(1, basis.depth), one / d);
            rpp::ops::VectorInplaceAdd<Strategy>{}(
                ctx, out_view, multiplier_view);
        }

        return out;
    }

    [[nodiscard]] static std::vector<Scalar> make_identity(Basis const& basis) {
        std::vector<Scalar> result(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> result_view(result.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorSetIdentity<Strategy>{}(ctx, result_view);
        return result;
    }
};

TEST_F(FreeTensorFMExpTests, MultipliesByExponentialOnTheRight) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const x = make_positive_degree_tensor('x', basis);
    auto const y = make_positive_degree_tensor('y', basis);

    auto const exp_x = apply_exp(basis, x);
    auto const exp_y = apply_exp(basis, y);

    auto const actual = apply_fmexp(basis, exp_x, y);
    auto const expected = apply_mul(basis, exp_x, exp_y);

    EXPECT_EQ(actual, expected);
}

TEST_F(FreeTensorFMExpTests, MatchesUntruncatedHornerDefinition) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const x = make_positive_degree_tensor('x', basis);
    auto const y = make_positive_degree_tensor('y', basis);
    auto const exp_x = apply_exp(basis, x);

    EXPECT_EQ(apply_fmexp(basis, exp_x, y),
              apply_fmexp_untruncated_horner(basis, exp_x, y));
}

TEST_F(FreeTensorFMExpTests, IdentityMultiplierIsExp) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const x = make_positive_degree_tensor('x', basis);
    auto const identity = make_identity(basis);

    EXPECT_EQ(apply_fmexp(basis, identity, x), apply_exp(basis, x));
}

TEST_F(FreeTensorFMExpTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const multiplier = Wrapper::make_batch('m', basis);
    auto const exponent = Wrapper::make_batch('x', basis);

    const auto err =
        rpp::ops::ft_fmexp(strategy,
                           {},
                           Wrapper::tensor_batch(actual, basis),
                           Wrapper::tensor_batch(multiplier, basis),
                           Wrapper::tensor_batch(exponent, basis),
                           basis,
                           Wrapper::tensor_count);
    EXPECT_TRUE(static_cast<bool>(err));

    Wrapper::apply_direct<rpp::ops::FTFMExp<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto mult = Wrapper::tensor_view(multiplier, basis, tensor_idx);
            auto exp = Wrapper::tensor_view(exponent, basis, tensor_idx);
            op(ctx, out, mult, exp);
        });

    EXPECT_EQ(actual, expected);
}

} // namespace
