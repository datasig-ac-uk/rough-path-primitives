#include <vector>
#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/st_adj_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/st_mul.hpp>
#include <rpp/cpu/single_thread/operations/basic/st_fma.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_pairing.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_st_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

template <typename Config>
class NumericShuffleTensorAdjointMulTests
    : public rpp::tests::TypedCpuShuffleTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuShuffleTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::ConstTensorView;
    using typename Base::Degree;
    using typename Base::Strategy;
    using typename Base::TensorView;
    using Base::const_tensor_view;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_tensor;
    using Base::make_unit_tensor;
    using Base::mutable_tensor_view;
    using Base::zero_tensor;

    static void expect_scalar_near(Accum actual, Accum expected) {
        auto const actual_d = static_cast<double>(actual);
        auto const expected_d = static_cast<double>(expected);
        auto const scale =
            std::max({1.0, std::abs(actual_d), std::abs(expected_d)});
        auto const tolerance =
            rpp::tests::NumericTolerance<typename Base::Scalar, Accum>::value *
            scale;
        EXPECT_NEAR(actual_d, expected_d, tolerance);
    }

    [[nodiscard]] static Accum pairing(Basis const& basis,
                                       std::vector<typename Base::Scalar> const& lhs,
                                       std::vector<typename Base::Scalar> const& rhs) {
        Accum result{0};
        ConstTensorView lhs_view(lhs.data(), basis);
        ConstTensorView rhs_view(rhs.data(), basis);

        auto const ctx = Base::make_context();
        rpp::ops::TensorPairing<Strategy>{}(ctx, result, lhs_view, rhs_view);
        return result;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_shuffle(Basis const& basis,
                std::vector<typename Base::Scalar> const& lhs,
                std::vector<typename Base::Scalar> const& rhs) {
        auto out = zero_tensor(basis);
        auto out_view = mutable_tensor_view(out, basis, full_range(basis));
        auto const lhs_view = const_tensor_view(lhs, basis, full_range(basis));
        auto const rhs_view = const_tensor_view(rhs, basis, full_range(basis));

        auto const ctx = Base::make_context();
        rpp::ops::STMul<Strategy>{}(ctx, out_view, lhs_view, rhs_view);
        return out;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_adj_mul(Basis const& basis,
                std::vector<typename Base::Scalar> const& op,
                std::vector<typename Base::Scalar> const& arg) {
        auto out = zero_tensor(basis);
        auto out_view = mutable_tensor_view(out, basis, full_range(basis));
        auto const op_view = const_tensor_view(op, basis, full_range(basis));
        auto const arg_view = const_tensor_view(arg, basis, full_range(basis));

        auto const ctx = Base::make_context();
        rpp::ops::STAdjMul<Strategy>{}(ctx, out_view, op_view, arg_view);
        return out;
    }
};

TYPED_TEST_SUITE(NumericShuffleTensorAdjointMulTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericShuffleTensorAdjointMulTests,
           SatisfiesAdjointPairingCriterion) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    auto const op = TestFixture::make_tensor(1, basis);
    auto const t = TestFixture::make_tensor(2, basis);
    auto const arg = TestFixture::make_tensor(3, basis);

    auto const product = TestFixture::run_shuffle(basis, op, t);
    auto const adjoint = TestFixture::run_adj_mul(basis, op, arg);

    TestFixture::expect_scalar_near(TestFixture::pairing(basis, product, arg),
                                    TestFixture::pairing(basis, t, adjoint));
}

TYPED_TEST(NumericShuffleTensorAdjointMulTests,
           IdentityOperatorReturnsArgumentForTruncatedView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    auto const op = TestFixture::make_unit_tensor(basis);
    auto const arg = TestFixture::make_tensor(7, basis);
    auto actual = TestFixture::zero_tensor(basis);

    typename TestFixture::TensorView out_view(actual.data(), basis);
    typename TestFixture::ConstTensorView op_view(
        op.data(),
        basis,
        typename TestFixture::Degree{0},
        typename TestFixture::Degree{0});
    typename TestFixture::ConstTensorView arg_view(arg.data(), basis);

    auto const ctx = TestFixture::make_context();
    rpp::ops::STAdjMul<typename TestFixture::Strategy>{}(
        ctx, out_view, op_view, arg_view);

    TestFixture::expect_tensor_near(actual, arg);
}

class ShuffleTensorAdjointMulTests : public testing::Test,
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
    apply_shuffle(Basis const& basis,
                  std::vector<Scalar> const& lhs,
                  std::vector<Scalar> const& rhs) {
        std::vector<Scalar> out(static_cast<std::size_t>(basis.size()));
        std::vector<Scalar> addend(static_cast<std::size_t>(basis.size()));

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> addend_view(addend.data(), basis);
        TensorView<Scalar const*> lhs_view(lhs.data(), basis);
        TensorView<Scalar const*> rhs_view(rhs.data(), basis);

        auto const ctx = make_context();
        rpp::ops::STFma<Strategy>{}(
            ctx, out_view, addend_view, lhs_view, rhs_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    apply_adj_mul(Basis const& basis,
                  std::vector<Scalar> const& op,
                  std::vector<Scalar> const& arg) {
        auto out = zero_tensor(basis);

        TensorView<Scalar*> out_view(out.data(), basis);
        TensorView<Scalar const*> op_view(op.data(), basis);
        TensorView<Scalar const*> arg_view(arg.data(), basis);

        auto const ctx = make_context();
        rpp::ops::STAdjMul<Strategy>{}(ctx, out_view, op_view, arg_view);
        return out;
    }

    [[nodiscard]] static std::vector<Scalar>
    make_identity_operator(Basis const& basis) {
        auto result = zero_tensor(basis);
        result[0] = one();
        return result;
    }
};

TEST_F(ShuffleTensorAdjointMulTests, SatisfiesAdjointPairingCriterion) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const a = make_tensor('a', basis);
    auto const b = make_tensor('b', basis);
    auto const x = make_tensor('x', basis);

    auto const product = apply_shuffle(basis, a, b);
    auto const adjoint = apply_adj_mul(basis, a, x);

    EXPECT_EQ(pairing(basis, product, x), pairing(basis, b, adjoint));
}

TEST_F(ShuffleTensorAdjointMulTests, IsBilinearInOperatorAndArgument) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const a1 = make_tensor('a', basis);
    auto const a2 = make_tensor('b', basis);
    auto const x1 = make_tensor('x', basis);
    auto const x2 = make_tensor('y', basis);

    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 1}});
    auto const gamma = make_scalar({{{{'r', 3}}, 5, 1}});
    auto const delta = make_scalar({{{{'s', 4}}, 7, 1}});

    auto const a = linear_combo(a1, alpha, a2, beta);
    auto const x = linear_combo(x1, gamma, x2, delta);

    auto const lhs = apply_adj_mul(basis, a, x);

    auto const ax11 = apply_adj_mul(basis, a1, x1);
    auto const ax12 = apply_adj_mul(basis, a1, x2);
    auto const ax21 = apply_adj_mul(basis, a2, x1);
    auto const ax22 = apply_adj_mul(basis, a2, x2);

    std::vector<Scalar> rhs(static_cast<std::size_t>(basis.size()));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] = (alpha * gamma) * ax11[i] + (alpha * delta) * ax12[i] +
            (beta * gamma) * ax21[i] + (beta * delta) * ax22[i];
    }

    EXPECT_EQ(lhs, rhs);
}

TEST_F(ShuffleTensorAdjointMulTests,
       IdentityOperatorReturnsArgumentForTruncatedView) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto const op = make_identity_operator(basis);
    auto const arg = make_tensor('x', basis);
    auto out = zero_tensor(basis);

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> op_view(op.data(), basis, Degree{0}, Degree{0});
    TensorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = make_context();
    rpp::ops::STAdjMul<Strategy>{}(ctx, out_view, op_view, arg_view);

    EXPECT_EQ(out, arg);
}

TEST_F(ShuffleTensorAdjointMulTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const op_arg = Wrapper::make_batch('a', basis);
    auto const arg = Wrapper::make_batch('x', basis);

    auto const err =
        rpp::ops::st_adj_mul(strategy,
                             typename Wrapper::Strategy::LaunchConfig{},
                             Wrapper::tensor_batch(actual, basis),
                             Wrapper::tensor_batch(op_arg, basis),
                             Wrapper::tensor_batch(arg, basis),
                             basis,
                             Wrapper::tensor_count);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::STAdjMul<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto operator_arg = Wrapper::tensor_view(op_arg, basis, tensor_idx);
            auto operand = Wrapper::tensor_view(arg, basis, tensor_idx);
            op(ctx, out, operator_arg, operand);
        });

    EXPECT_EQ(actual, expected);
}

} // namespace
