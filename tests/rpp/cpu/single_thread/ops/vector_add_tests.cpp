#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_add.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_vector_ops_test_helper.hpp"

namespace {

template <typename Config>
class NumericVectorAddTests : public rpp::tests::TypedCpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuVectorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::linear_combo;
    using Base::make_tensor;
    using Base::mutable_vector_view;
    using Base::const_vector_view;
    using Base::zero_tensor;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    reference_add(std::vector<typename Base::Scalar> const& initial_out,
                  std::vector<typename Base::Scalar> const& lhs,
                  std::vector<typename Base::Scalar> const& rhs,
                  Basis const& basis,
                  DegreeRange out_range,
                  DegreeRange lhs_range,
                  DegreeRange rhs_range,
                  Accum alpha,
                  Accum beta) {
        auto result = initial_out;
        auto const min_degree =
            std::max({out_range.min, lhs_range.min, rhs_range.min});
        auto const max_degree =
            std::min({out_range.max, lhs_range.max, rhs_range.max});
        if (max_degree < min_degree) {
            return result;
        }
        for (auto idx = basis.start_of_degree(min_degree);
             idx < basis.end_of_degree(max_degree);
             ++idx) {
            auto const i = static_cast<std::size_t>(idx);
            result[i] = static_cast<typename Base::Scalar>(
                alpha * static_cast<Accum>(lhs[i]) +
                beta * static_cast<Accum>(rhs[i]));
        }
        return result;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_add(std::vector<typename Base::Scalar> const& initial_out,
            std::vector<typename Base::Scalar> const& lhs,
            std::vector<typename Base::Scalar> const& rhs,
            Basis const& basis,
            DegreeRange out_range,
            DegreeRange lhs_range,
            DegreeRange rhs_range,
            Accum alpha,
            Accum beta) {
        auto actual = initial_out;
        auto out_view = mutable_vector_view(actual, basis, out_range);
        auto const lhs_view = const_vector_view(lhs, basis, lhs_range);
        auto const rhs_view = const_vector_view(rhs, basis, rhs_range);
        auto const ctx = Base::make_context();
        rpp::ops::VectorAdd<Strategy>{}(ctx, out_view, lhs_view, rhs_view, alpha, beta);
        return actual;
    }
};

TYPED_TEST_SUITE(NumericVectorAddTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericVectorAddTests, MatchesReferenceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(1, basis);
    auto const lhs = TestFixture::make_tensor(2, basis);
    auto const rhs = TestFixture::make_tensor(3, basis);
    auto const alpha = typename TestFixture::Accum{0.75};
    auto const beta = typename TestFixture::Accum{-1.25};

    auto const actual = TestFixture::run_add(
        out,
        lhs,
        rhs,
        basis,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        alpha,
        beta);
    auto const expected = TestFixture::reference_add(
        out,
        lhs,
        rhs,
        basis,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        alpha,
        beta);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorAddTests, EqualsLinearCombinationWhenOutputStartsZero) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const lhs = TestFixture::make_tensor(4, basis);
    auto const rhs = TestFixture::make_tensor(5, basis);
    auto const alpha = typename TestFixture::Accum{1.5};
    auto const beta = typename TestFixture::Accum{0.5};
    auto const zero = TestFixture::zero_tensor(basis);

    auto const actual = TestFixture::run_add(
        zero,
        lhs,
        rhs,
        basis,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        alpha,
        beta);
    auto const expected = TestFixture::linear_combo(lhs, alpha, rhs, beta);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorAddTests, RespectsTruncatedIntersection) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(6, basis);
    auto const lhs = TestFixture::make_tensor(7, basis);
    auto const rhs = TestFixture::make_tensor(8, basis);
    auto const alpha = typename TestFixture::Accum{-0.5};
    auto const beta = typename TestFixture::Accum{2.0};
    typename TestFixture::DegreeRange const out_range{1, 3};
    typename TestFixture::DegreeRange const lhs_range{2, TestFixture::depth};
    typename TestFixture::DegreeRange const rhs_range{0, 2};

    auto const actual = TestFixture::run_add(
        out, lhs, rhs, basis, out_range, lhs_range, rhs_range, alpha, beta);
    auto const expected = TestFixture::reference_add(
        out, lhs, rhs, basis, out_range, lhs_range, rhs_range, alpha, beta);
    TestFixture::expect_tensor_near(actual, expected);
}

} // namespace
