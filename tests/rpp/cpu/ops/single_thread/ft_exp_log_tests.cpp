#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/ops/single_thread/ft_exp.hpp>
#include <rpp/cpu/ops/single_thread/ft_inplace_mul.hpp>
#include <rpp/cpu/ops/single_thread/ft_log.hpp>
#include <rpp/cpu/ops/single_thread/tensor_add_identity.hpp>
#include <rpp/cpu/ops/single_thread/tensor_set_identity.hpp>
#include <rpp/cpu/ops/single_thread/vector_set_zero.hpp>
#include <rpp/dense/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class FreeTensorExpLogTests
    : public testing::Test,
      public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 2;
    static constexpr Degree depth = 4;

    [[nodiscard]] static std::vector<Scalar> make_positive_degree_tensor(
        char marker,
        Basis const& basis
    )
    {
        auto result = make_tensor(marker, basis);
        result[0] = Scalar{0};
        return result;
    }

    [[nodiscard]] static std::vector<Scalar> apply_exp(
        Basis const& basis,
        std::vector<Scalar> const& arg
    )
    {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::FTExp<Strategy>{}(ctx, out_view, arg_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar> apply_log(
        Basis const& basis,
        std::vector<Scalar> const& arg
    )
    {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::FTLog<Strategy>{}(ctx, out_view, arg_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar> apply_exp_untruncated_horner(
        Basis const& basis,
        std::vector<Scalar> const& arg
    )
    {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::TensorSetIdentity<Strategy>{}(ctx, out_view);

        Scalar const one{1};
        for (Degree d = basis.depth; d > 0; --d) {
            rpp::ops::FTInplaceMul<Strategy>{}(
                ctx,
                out_view,
                arg_view.truncate(1, basis.depth),
                one / d
            );
            rpp::ops::TensorAddIdentity<Strategy>{}(ctx, out_view);
        }

        return out;
    }

    [[nodiscard]] static std::vector<Scalar> apply_log_untruncated_horner(
        Basis const& basis,
        std::vector<Scalar> const& arg
    )
    {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::VectorSetZero<Strategy>{}(ctx, out_view);

        Scalar const one{1};
        for (Degree d = basis.depth; d > 0; --d) {
            auto const coefficient = (d % 2 == 0 ? -one : one) / d;
            rpp::ops::TensorAddIdentity<Strategy>{}(ctx, out_view, coefficient);
            rpp::ops::FTInplaceMul<Strategy>{}(
                ctx,
                out_view,
                arg_view.truncate(1, basis.depth)
            );
        }

        return out;
    }
};

TEST_F(FreeTensorExpLogTests, LogExpRoundTripForPositiveDegreeInput)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const x = make_positive_degree_tensor('x', basis);

    auto const exp_x = apply_exp(basis, x);
    auto const log_exp_x = apply_log(basis, exp_x);

    EXPECT_EQ(log_exp_x, x);
}

TEST_F(FreeTensorExpLogTests, ExpLogRoundTripForExponentialInput)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const x = make_positive_degree_tensor('x', basis);
    auto const exp_x = apply_exp(basis, x);

    auto const log_exp_x = apply_log(basis, exp_x);
    auto const exp_log_exp_x = apply_exp(basis, log_exp_x);

    EXPECT_EQ(exp_log_exp_x, exp_x);
}

TEST_F(FreeTensorExpLogTests, ExpMatchesUntruncatedHornerDefinition)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const x = make_positive_degree_tensor('x', basis);

    EXPECT_EQ(apply_exp(basis, x), apply_exp_untruncated_horner(basis, x));
}

TEST_F(FreeTensorExpLogTests, LogMatchesUntruncatedHornerDefinition)
{
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const x = make_positive_degree_tensor('x', basis);
    auto const exp_x = apply_exp(basis, x);

    EXPECT_EQ(apply_log(basis, exp_x), apply_log_untruncated_horner(basis, exp_x));
}

TEST_F(FreeTensorExpLogTests, ExpKernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('x', basis);

    rpp::cpu::single_thread::ft_exp_kernel(
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(arg, basis),
        basis,
        strategy,
        Wrapper::tensor_count
    );
    Wrapper::apply_direct<rpp::ops::FTExp<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto operand = Wrapper::tensor_view(arg, basis, tensor_idx);
            op(ctx, out, operand);
        }
    );

    EXPECT_EQ(actual, expected);
}

TEST_F(FreeTensorExpLogTests, LogKernelWrapperMatchesDirectOperation)
{
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('x', basis);

    rpp::cpu::single_thread::ft_log_kernel(
        Wrapper::tensor_batch(actual, basis),
        Wrapper::tensor_batch(arg, basis),
        basis,
        strategy,
        Wrapper::tensor_count
    );
    Wrapper::apply_direct<rpp::ops::FTLog<Wrapper::Strategy>>(
        basis,
        [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto operand = Wrapper::tensor_view(arg, basis, tensor_idx);
            op(ctx, out, operand);
        }
    );

    EXPECT_EQ(actual, expected);
}

} // namespace
