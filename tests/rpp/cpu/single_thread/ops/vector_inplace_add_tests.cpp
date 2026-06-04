#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_inplace_add.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_vector_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class VectorInplaceAddTests : public testing::Test,
                              public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 3;
};

TEST_F(VectorInplaceAddTests, AddsRhsCoefficientwise) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);

    VectorView<Scalar*> lhs_view(lhs.data(), basis);
    VectorView<Scalar const*> rhs_view(rhs.data(), basis);

    auto const ctx = make_context();
    rpp::ops::VectorInplaceAdd<Strategy>{}(ctx, lhs_view, rhs_view);

    std::vector<Scalar> expected;
    expected.reserve(static_cast<std::size_t>(basis.size()));
    for_each_index(
        basis, [&expected, &basis](Degree degree, Index level_index) {
            expected.emplace_back(symbol('a', basis, degree, level_index) +
                                  symbol('b', basis, degree, level_index));
        });

    EXPECT_EQ(lhs, expected);
}

TEST_F(VectorInplaceAddTests, AddsScaledRhsCoefficientwise) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);
    auto const multiplier = make_scalar({{{{'c', 0}}, 1, 1}});

    VectorView<Scalar*> lhs_view(lhs.data(), basis);
    VectorView<Scalar const*> rhs_view(rhs.data(), basis);

    auto const ctx = make_context();
    rpp::ops::VectorInplaceAdd<Strategy>{}(ctx, lhs_view, rhs_view, multiplier);

    std::vector<Scalar> expected;
    expected.reserve(static_cast<std::size_t>(basis.size()));
    for_each_index(
        basis,
        [&expected, &basis, &multiplier](Degree degree, Index level_index) {
            expected.emplace_back(symbol('a', basis, degree, level_index) +
                                  multiplier *
                                      symbol('b', basis, degree, level_index));
        });

    EXPECT_EQ(lhs, expected);
}

TEST_F(VectorInplaceAddTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const alpha = Wrapper::make_scalar({{{{'p', 1}}, 7, 5}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const rhs = Wrapper::make_batch('b', basis);

    auto const err =
        rpp::ops::vector_inplace_add(strategy,
                                     typename Wrapper::Strategy::LaunchConfig{},
                                     Wrapper::vector_batch(actual, basis),
                                     Wrapper::vector_batch(rhs, basis),
                                     basis,
                                     Wrapper::tensor_count,
                                     alpha);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::VectorInplaceAdd<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto lhs = Wrapper::vector_view(expected, basis, tensor_idx);
            auto arg = Wrapper::vector_view(rhs, basis, tensor_idx);
            op(ctx, lhs, arg, alpha);
        });

    EXPECT_EQ(actual, expected);
}

template <typename Config>
class NumericVectorInplaceAddTests
    : public rpp::tests::TypedCpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuVectorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::ConstVectorView;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using typename Base::VectorView;
    using Base::const_vector_view;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_tensor;
    using Base::mutable_vector_view;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    reference_inplace_add(std::vector<typename Base::Scalar> const& lhs,
                          std::vector<typename Base::Scalar> const& rhs,
                          Basis const& basis,
                          DegreeRange lhs_range,
                          DegreeRange rhs_range,
                          Accum alpha) {
        auto result = lhs;
        auto const min_degree = std::max(lhs_range.min, rhs_range.min);
        auto const max_degree = std::min(lhs_range.max, rhs_range.max);
        if (max_degree < min_degree) {
            return result;
        }
        for (auto idx = basis.start_of_degree(min_degree);
             idx < basis.end_of_degree(max_degree);
             ++idx) {
            auto const i = static_cast<std::size_t>(idx);
            result[i] = static_cast<typename Base::Scalar>(
                static_cast<Accum>(result[i]) + alpha * static_cast<Accum>(rhs[i]));
        }
        return result;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_inplace_add(Basis const& basis,
                    std::vector<typename Base::Scalar> const& lhs,
                    std::vector<typename Base::Scalar> const& rhs,
                    DegreeRange lhs_range,
                    DegreeRange rhs_range,
                    Accum alpha) {
        auto actual = lhs;
        auto lhs_view = mutable_vector_view(actual, basis, lhs_range);
        auto const rhs_view = const_vector_view(rhs, basis, rhs_range);

        auto const ctx = Base::make_context();
        rpp::ops::VectorInplaceAdd<Strategy>{}(ctx, lhs_view, rhs_view, alpha);
        return actual;
    }
};

TYPED_TEST_SUITE(NumericVectorInplaceAddTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericVectorInplaceAddTests, MatchesReferenceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const lhs = TestFixture::make_tensor(1, basis);
    auto const rhs = TestFixture::make_tensor(2, basis);
    auto const alpha = typename TestFixture::Accum{-1.75};

    auto const actual = TestFixture::run_inplace_add(
        basis, lhs, rhs, TestFixture::full_range(basis), TestFixture::full_range(basis), alpha);
    auto const expected = TestFixture::reference_inplace_add(
        lhs, rhs, basis, TestFixture::full_range(basis), TestFixture::full_range(basis), alpha);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorInplaceAddTests, AlphaZeroIsNoOp) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const lhs = TestFixture::make_tensor(3, basis);
    auto const rhs = TestFixture::make_tensor(4, basis);

    auto const actual = TestFixture::run_inplace_add(
        basis,
        lhs,
        rhs,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        typename TestFixture::Accum{0});
    TestFixture::expect_tensor_near(actual, lhs);
}

TYPED_TEST(NumericVectorInplaceAddTests, RespectsTruncatedIntersection) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const lhs = TestFixture::make_tensor(5, basis);
    auto const rhs = TestFixture::make_tensor(6, basis);
    auto const alpha = typename TestFixture::Accum{0.625};
    typename TestFixture::DegreeRange const lhs_range{1, TestFixture::depth};
    typename TestFixture::DegreeRange const rhs_range{2, TestFixture::depth};

    auto const actual = TestFixture::run_inplace_add(
        basis, lhs, rhs, lhs_range, rhs_range, alpha);
    auto const expected =
        TestFixture::reference_inplace_add(lhs, rhs, basis, lhs_range, rhs_range, alpha);
    TestFixture::expect_tensor_near(actual, expected);
}

} // namespace
