#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/linalg/vector_assign.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_vector_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class VectorAssignTests : public testing::Test,
                          public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;
};

TEST_F(VectorAssignTests, CopiesSourceCoefficientwise) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('a', basis);
    auto const arg = make_tensor('b', basis);

    VectorView<Scalar*> out_view(out.data(), basis);
    VectorView<Scalar const*> arg_view(arg.data(), basis);

    auto const ctx = make_context();
    rpp::ops::VectorAssign<Strategy>{}(ctx, out_view, arg_view);

    EXPECT_EQ(out, arg);
}

TEST_F(VectorAssignTests, CopiesOverlapAndZerosLowerGap) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('a', basis);
    auto const arg = make_tensor('b', basis);
    auto expected = out;

    for (Index i = basis.start_of_degree(1); i < basis.end_of_degree(1); ++i) {
        expected[static_cast<std::size_t>(i)] = Scalar{};
    }

    for (Degree degree = 2; degree <= 3; ++degree) {
        auto const begin = basis.start_of_degree(degree);
        auto const end = basis.end_of_degree(degree);
        for (Index i = begin; i < end; ++i) {
            expected[static_cast<std::size_t>(i)] =
                arg[static_cast<std::size_t>(i)];
        }
    }

    VectorView<Scalar*> out_view(out.data(), basis, 1, 3);
    VectorView<Scalar const*> arg_view(arg.data(), basis, 2, 4);

    auto const ctx = make_context();
    rpp::ops::VectorAssign<Strategy>{}(ctx, out_view, arg_view);

    EXPECT_EQ(out, expected);
}

TEST_F(VectorAssignTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const arg = Wrapper::make_batch('b', basis);

    auto const err =
        rpp::ops::vector_assign(strategy,
                                typename Wrapper::Strategy::LaunchConfig{},
                                Wrapper::vector_batch(actual, basis),
                                Wrapper::vector_batch(arg, basis),
                                basis,
                                Wrapper::tensor_count);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::VectorAssign<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::vector_view(expected, basis, tensor_idx);
            auto rhs = Wrapper::vector_view(arg, basis, tensor_idx);
            op(ctx, out, rhs);
        });

    EXPECT_EQ(actual, expected);
}

template <typename Config>
class NumericVectorAssignTests
    : public rpp::tests::TypedCpuVectorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuVectorOpTestBase<Config>;
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
    reference_assign(std::vector<typename Base::Scalar> const& out,
                     std::vector<typename Base::Scalar> const& arg,
                     Basis const& basis,
                     DegreeRange out_range,
                     DegreeRange arg_range) {
        auto result = out;
        for (auto degree = out_range.min; degree <= out_range.max; ++degree) {
            const bool has_arg = arg_range.min <= degree && degree <= arg_range.max;
            for (auto idx = basis.start_of_degree(degree);
                 idx < basis.end_of_degree(degree); ++idx) {
                const auto i = static_cast<std::size_t>(idx);
                result[i] = has_arg ? arg[i] : typename Base::Scalar{};
            }
        }
        return result;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_assign(Basis const& basis,
               std::vector<typename Base::Scalar> const& out,
               std::vector<typename Base::Scalar> const& arg,
               DegreeRange out_range,
               DegreeRange arg_range) {
        auto actual = out;
        auto out_view = mutable_vector_view(actual, basis, out_range);
        auto const arg_view = const_vector_view(arg, basis, arg_range);

        auto const ctx = Base::make_context();
        rpp::ops::VectorAssign<Strategy>{}(ctx, out_view, arg_view);
        return actual;
    }
};

TYPED_TEST_SUITE(NumericVectorAssignTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericVectorAssignTests, CopiesSourceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(1, basis);
    auto const arg = TestFixture::make_tensor(2, basis);

    auto const actual = TestFixture::run_assign(
        basis, out, arg, TestFixture::full_range(basis), TestFixture::full_range(basis));
    TestFixture::expect_tensor_near(actual, arg);
}

TYPED_TEST(NumericVectorAssignTests, CopiesOverlapAndZerosLowerGap) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(3, basis);
    auto const arg = TestFixture::make_tensor(4, basis);
    typename TestFixture::DegreeRange const out_range{1, 3};
    typename TestFixture::DegreeRange const arg_range{2, TestFixture::depth};

    auto const actual = TestFixture::run_assign(basis, out, arg, out_range, arg_range);
    auto const expected =
        TestFixture::reference_assign(out, arg, basis, out_range, arg_range);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorAssignTests, UnitOnlyArgumentZerosAllHigherDegrees) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(7, basis);
    auto const arg = TestFixture::make_tensor(8, basis);
    auto expected = TestFixture::zero_tensor(basis);
    expected[0] = arg[0];

    auto const actual = TestFixture::run_assign(
        basis, out, arg, TestFixture::full_range(basis),
        typename TestFixture::DegreeRange{0, 0});

    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorAssignTests,
           ShorterArgumentZerosTailOfTruncatedOutputOnly) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(9, basis);
    auto const arg = TestFixture::make_tensor(10, basis);
    auto expected = out;

    // Copy degrees 1 and 2, clear degree 3, preserve degrees 0 and 4.
    for (auto i = basis.start_of_degree(1); i < basis.end_of_degree(2); ++i) {
        expected[static_cast<std::size_t>(i)] = arg[static_cast<std::size_t>(i)];
    }
    for (auto i = basis.start_of_degree(3); i < basis.end_of_degree(3); ++i) {
        expected[static_cast<std::size_t>(i)] = typename TestFixture::Scalar{0};
    }

    auto const actual = TestFixture::run_assign(
        basis, out, arg, typename TestFixture::DegreeRange{1, 3},
        typename TestFixture::DegreeRange{1, 2});

    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorAssignTests, ZerosBothGapsAroundSingleDegreeArgument) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(11, basis);
    auto const arg = TestFixture::make_tensor(12, basis);
    typename TestFixture::DegreeRange const out_range{1, 3};
    typename TestFixture::DegreeRange const arg_range{2, 2};

    auto const actual = TestFixture::run_assign(basis, out, arg, out_range, arg_range);
    auto const expected =
        TestFixture::reference_assign(out, arg, basis, out_range, arg_range);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericVectorAssignTests, NoOverlapClearsOnlyOutputView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const out = TestFixture::make_tensor(5, basis);
    auto const arg = TestFixture::make_tensor(6, basis);
    using DegreeRange = typename TestFixture::DegreeRange;
    for (bool arg_above_output : {false, true}) {
        SCOPED_TRACE(arg_above_output ? "argument above output" : "argument below output");
        const DegreeRange out_range = arg_above_output
            ? DegreeRange{0, 0} : DegreeRange{2, 3};
        const DegreeRange arg_range = arg_above_output
            ? DegreeRange{1, TestFixture::depth} : DegreeRange{0, 1};

        auto const actual = TestFixture::run_assign(basis, out, arg, out_range, arg_range);
        auto const expected =
            TestFixture::reference_assign(out, arg, basis, out_range, arg_range);
        TestFixture::expect_tensor_near(actual, expected);
    }
}

} // namespace
