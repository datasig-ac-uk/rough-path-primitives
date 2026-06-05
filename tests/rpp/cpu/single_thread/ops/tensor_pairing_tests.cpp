#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/tensor_pairing.hpp>
#include <rpp/views/scalar_view.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"

namespace {

template <typename Config>
class NumericTensorPairingTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::ConstTensorView;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using Base::const_tensor_view;
    using Base::full_range;
    using Base::linear_combo;
    using Base::make_tensor;

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

    [[nodiscard]] static Accum reference_pairing(
        Basis const& basis,
        std::vector<typename Base::Scalar> const& functional,
        std::vector<typename Base::Scalar> const& arg,
        DegreeRange functional_range,
        DegreeRange arg_range) {
        auto const min_degree =
            std::max(functional_range.min, arg_range.min);
        auto const max_degree =
            std::min(functional_range.max, arg_range.max);
        if (max_degree < min_degree) {
            return Accum{0};
        }

        Accum result{0};
        for (auto idx = basis.start_of_degree(min_degree);
             idx < basis.end_of_degree(max_degree);
             ++idx) {
            result += static_cast<Accum>(
                          functional[static_cast<std::size_t>(idx)]) *
                static_cast<Accum>(arg[static_cast<std::size_t>(idx)]);
        }
        return result;
    }

    [[nodiscard]] static Accum run_pairing(
        Basis const& basis,
        std::vector<typename Base::Scalar> const& functional,
        std::vector<typename Base::Scalar> const& arg,
        DegreeRange functional_range,
        DegreeRange arg_range) {
        Accum out{0};
        ConstTensorView functional_view(
            functional.data(), basis, functional_range.min, functional_range.max);
        ConstTensorView arg_view(arg.data(), basis, arg_range.min, arg_range.max);

        auto const ctx = Base::make_context();
        rpp::ops::TensorPairing<Strategy>{}(ctx, out, functional_view, arg_view);
        return out;
    }
};

TYPED_TEST_SUITE(NumericTensorPairingTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericTensorPairingTests, IsSymmetric) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    auto const lhs = TestFixture::make_tensor(1, basis);
    auto const rhs = TestFixture::make_tensor(2, basis);

    auto const lhs_rhs = TestFixture::run_pairing(
        basis, lhs, rhs, TestFixture::full_range(basis), TestFixture::full_range(basis));
    auto const rhs_lhs = TestFixture::run_pairing(
        basis, rhs, lhs, TestFixture::full_range(basis), TestFixture::full_range(basis));

    TestFixture::expect_scalar_near(lhs_rhs, rhs_lhs);
}

TYPED_TEST(NumericTensorPairingTests, IsLinearInEachArgument) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const alpha = typename TestFixture::Accum{0.75};
    auto const beta = typename TestFixture::Accum{-1.25};

    auto const x = TestFixture::make_tensor(3, basis);
    auto const y = TestFixture::make_tensor(4, basis);
    auto const z = TestFixture::make_tensor(5, basis);
    auto const combo = TestFixture::linear_combo(x, alpha, y, beta);

    auto const lhs = TestFixture::run_pairing(
        basis, combo, z, TestFixture::full_range(basis), TestFixture::full_range(basis));
    auto const expected_lhs =
        alpha * TestFixture::run_pairing(
                    basis,
                    x,
                    z,
                    TestFixture::full_range(basis),
                    TestFixture::full_range(basis)) +
        beta * TestFixture::run_pairing(
                   basis,
                   y,
                   z,
                   TestFixture::full_range(basis),
                   TestFixture::full_range(basis));
    TestFixture::expect_scalar_near(lhs, expected_lhs);

    auto const rhs = TestFixture::run_pairing(
        basis, z, combo, TestFixture::full_range(basis), TestFixture::full_range(basis));
    auto const expected_rhs =
        alpha * TestFixture::run_pairing(
                    basis,
                    z,
                    x,
                    TestFixture::full_range(basis),
                    TestFixture::full_range(basis)) +
        beta * TestFixture::run_pairing(
                   basis,
                   z,
                   y,
                   TestFixture::full_range(basis),
                   TestFixture::full_range(basis));
    TestFixture::expect_scalar_near(rhs, expected_rhs);
}

TYPED_TEST(NumericTensorPairingTests, RespectsTruncatedOperandViews) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    auto const functional = TestFixture::make_tensor(6, basis);
    auto const arg = TestFixture::make_tensor(7, basis);

    typename TestFixture::DegreeRange const functional_range{1, TestFixture::depth};
    typename TestFixture::DegreeRange const arg_range{
        static_cast<typename TestFixture::Degree>(
            std::min<typename TestFixture::Degree>(2, TestFixture::depth)),
        TestFixture::depth};

    auto const actual = TestFixture::run_pairing(
        basis, functional, arg, functional_range, arg_range);
    auto const expected = TestFixture::reference_pairing(
        basis, functional, arg, functional_range, arg_range);
    TestFixture::expect_scalar_near(actual, expected);

    if (basis.depth >= 1) {
        typename TestFixture::DegreeRange const disjoint_functional{0, 0};
        typename TestFixture::DegreeRange const disjoint_arg{1, basis.depth};
        auto const zero = TestFixture::run_pairing(
            basis, functional, arg, disjoint_functional, disjoint_arg);
        TestFixture::expect_scalar_near(zero, typename TestFixture::Accum{0});
    }
}

TYPED_TEST(NumericTensorPairingTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = typename TestFixture::Strategy{};

    auto const functional = TestFixture::make_tensor(8, basis);
    auto const arg = TestFixture::make_tensor(9, basis);
    std::vector<typename TestFixture::Accum> actual(1);
    std::vector<typename TestFixture::Accum> expected(1);

    auto const err = rpp::ops::tensor_pairing(
        strategy,
        typename TestFixture::Strategy::LaunchConfig{},
        rpp::make_scalar_batch(actual.data()),
        rpp::make_tensor_batch(functional.data(), basis.size(), 0, basis.depth),
        rpp::make_tensor_batch(arg.data(), basis.size(), 0, basis.depth),
        basis,
        1);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();

    typename TestFixture::ConstTensorView functional_view(functional.data(), basis);
    typename TestFixture::ConstTensorView arg_view(arg.data(), basis);
    auto const ctx = TestFixture::Strategy::make_context(nullptr);
    rpp::ops::TensorPairing<typename TestFixture::Strategy>{}(
        ctx, expected[0], functional_view, arg_view);

    TestFixture::expect_scalar_near(actual[0], expected[0]);
}

} // namespace
