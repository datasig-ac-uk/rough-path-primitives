#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_scalar_multiply.hpp>

#include "cpu_typed_vector_ops_test_helper.hpp"

namespace {

template <typename Config>
class NumericVectorScalarMultiplyTests
    : public rpp::tests::TypedCpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuVectorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_tensor;
    using Base::mutable_vector_view;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    reference_scalar_multiply(std::vector<typename Base::Scalar> const& vec,
                              Basis const& basis,
                              DegreeRange range,
                              Accum scalar) {
        auto result = vec;
        for (auto idx = basis.start_of_degree(range.min);
             idx < basis.end_of_degree(range.max);
             ++idx) {
            auto const i = static_cast<std::size_t>(idx);
            result[i] =
                static_cast<typename Base::Scalar>(static_cast<Accum>(result[i]) * scalar);
        }
        return result;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_scalar_multiply(std::vector<typename Base::Scalar> const& vec,
                        Basis const& basis,
                        DegreeRange range,
                        Accum scalar) {
        auto actual = vec;
        auto vec_view = mutable_vector_view(actual, basis, range);
        auto const ctx = Base::make_context();
        rpp::ops::VectorScalarMultiply<Strategy>{}(ctx, vec_view, scalar);
        return actual;
    }
};

TYPED_TEST_SUITE(NumericVectorScalarMultiplyTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericVectorScalarMultiplyTests, MatchesReferenceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const vec = TestFixture::make_tensor(1, basis);
    auto const scalar = typename TestFixture::Accum{-1.125};

    auto const actual = TestFixture::run_scalar_multiply(
        vec, basis, TestFixture::full_range(basis), scalar);
    auto const expected = TestFixture::reference_scalar_multiply(
        vec, basis, TestFixture::full_range(basis), scalar);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorScalarMultiplyTests, ZeroScalarZerosActiveSlice) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const vec = TestFixture::make_tensor(2, basis);
    typename TestFixture::DegreeRange const range{0, 1};

    auto const actual = TestFixture::run_scalar_multiply(
        vec, basis, range, typename TestFixture::Accum{0});
    auto const expected = TestFixture::reference_scalar_multiply(
        vec, basis, range, typename TestFixture::Accum{0});
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorScalarMultiplyTests, RespectsTruncatedView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const vec = TestFixture::make_tensor(3, basis);
    auto const scalar = typename TestFixture::Accum{0.625};
    typename TestFixture::DegreeRange const range{1, TestFixture::depth};

    auto const actual = TestFixture::run_scalar_multiply(vec, basis, range, scalar);
    auto const expected = TestFixture::reference_scalar_multiply(vec, basis, range, scalar);
    TestFixture::expect_tensor_near(actual, expected);
}

} // namespace
