#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/st_fma.hpp>
#include <rpp/cpu/single_thread/operations/basic/st_inplace_fma.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_st_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

using Helper = rpp::tests::PolynomialTensorHelper;
using Degree = Helper::Degree;
using Scalar = Helper::Scalar;

struct DegreeRange {
    Degree min;
    Degree max;
};

struct InplaceFmaViewCase {
    char const* name;
    DegreeRange a;
    DegreeRange b;
    DegreeRange c;
};

class ShuffleTensorInplaceFmaTests : public testing::Test, public Helper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static TensorView<Scalar*> mutable_tensor_view(
        std::vector<Scalar>& data, Basis const& basis, DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }

    [[nodiscard]] static TensorView<Scalar const*>
    const_tensor_view(std::vector<Scalar> const& data,
                      Basis const& basis,
                      DegreeRange range) {
        return {data.data(), basis, range.min, range.max};
    }

    static void
    expect_inplace_fma_matches_out_of_place(DegreeRange a_range,
                                            DegreeRange b_range,
                                            DegreeRange c_range,
                                            Scalar const& alpha = Scalar{1},
                                            Scalar const& beta = Scalar{1}) {
        auto const basis_data = BasisData(width, depth);
        auto const& basis = basis_data.basis;

        auto const initial_a = make_tensor('a', basis);
        auto inplace_a = initial_a;
        auto expected = initial_a;
        auto const b = make_tensor('b', basis);
        auto const c = make_tensor('c', basis);

        auto inplace_a_view = mutable_tensor_view(inplace_a, basis, a_range);
        auto expected_view = mutable_tensor_view(expected, basis, a_range);
        auto const initial_a_view =
            const_tensor_view(initial_a, basis, a_range);
        auto const b_view = const_tensor_view(b, basis, b_range);
        auto const c_view = const_tensor_view(c, basis, c_range);

        auto const ctx = make_context();
        rpp::ops::STInplaceFma<Strategy>{}(
            ctx, inplace_a_view, b_view, c_view, alpha, beta);
        rpp::ops::STFma<Strategy>{}(
            ctx, expected_view, initial_a_view, b_view, c_view, alpha, beta);

        EXPECT_EQ(inplace_a, expected);
    }
};

TEST_F(ShuffleTensorInplaceFmaTests, MatchesOutOfPlaceFmaForFullViews) {
    expect_inplace_fma_matches_out_of_place({0, depth}, {0, depth}, {0, depth});
}

TEST_F(ShuffleTensorInplaceFmaTests, MatchesOutOfPlaceFmaForTruncatedInputs) {
    InplaceFmaViewCase const cases[] = {
        {"a has positive min degree", {2, depth}, {0, depth}, {0, depth}},
        {"a has truncated max degree", {0, 2}, {0, depth}, {0, depth}},
        {"b has positive min degree", {0, depth}, {1, depth}, {0, depth}},
        {"c has positive min degree", {0, depth}, {0, depth}, {1, depth}},
        {"b and c have truncated max degrees", {0, depth}, {0, 2}, {0, 1}},
        {"all views are interior ranges", {1, 3}, {1, 2}, {1, 3}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_inplace_fma_matches_out_of_place(
            test_case.a, test_case.b, test_case.c);
    }
}

TEST_F(ShuffleTensorInplaceFmaTests,
       MatchesOutOfPlaceFmaWithScaledAddendAndProduct) {
    auto const alpha = make_scalar({{{{'p', 1}}, 3, 2}});
    auto const beta = make_scalar({{{{'q', 2}}, 5, 3}});

    expect_inplace_fma_matches_out_of_place(
        {1, depth}, {0, 2}, {1, depth}, alpha, beta);
}

TEST_F(ShuffleTensorInplaceFmaTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const alpha = Wrapper::make_scalar({{{{'p', 1}}, 3, 2}});
    auto const beta = Wrapper::make_scalar({{{{'q', 2}}, 5, 3}});

    auto actual = Wrapper::make_batch('a', basis);
    auto expected = actual;
    auto const b = Wrapper::make_batch('b', basis);
    auto const c = Wrapper::make_batch('c', basis);

    auto const err =
        rpp::ops::st_inplace_fma(strategy,
                                 typename Wrapper::Strategy::LaunchConfig{},
                                 Wrapper::tensor_batch(actual, basis),
                                 Wrapper::tensor_batch(b, basis),
                                 Wrapper::tensor_batch(c, basis),
                                 basis,
                                 Wrapper::tensor_count,
                                 alpha,
                                 beta);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::STInplaceFma<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto a = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto left = Wrapper::tensor_view(b, basis, tensor_idx);
            auto right = Wrapper::tensor_view(c, basis, tensor_idx);
            op(ctx, a, left, right, alpha, beta);
        });

    EXPECT_EQ(actual, expected);
}

template <typename Config>
class NumericShuffleTensorInplaceFmaTests
    : public rpp::tests::TypedCpuShuffleTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuShuffleTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::Basis;
    using typename Base::ConstTensorView;
    using typename Base::DegreeRange;
    using typename Base::Strategy;
    using typename Base::TensorView;
    using Base::const_tensor_view;
    using Base::expect_tensor_near;
    using Base::full_range;
    using Base::make_tensor;
    using Base::mutable_tensor_view;
    using Base::reference_fma;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_inplace_fma(Basis const& basis,
                    std::vector<typename Base::Scalar> const& initial_a,
                    std::vector<typename Base::Scalar> const& b,
                    std::vector<typename Base::Scalar> const& c,
                    DegreeRange a_range,
                    DegreeRange b_range,
                    DegreeRange c_range,
                    Accum alpha = Accum{1},
                    Accum beta = Accum{1}) {
        auto a = initial_a;
        auto a_view = mutable_tensor_view(a, basis, a_range);
        auto const b_view = const_tensor_view(b, basis, b_range);
        auto const c_view = const_tensor_view(c, basis, c_range);

        auto const ctx = Base::make_context();
        rpp::ops::STInplaceFma<Strategy>{}(
            ctx, a_view, b_view, c_view, alpha, beta);
        return a;
    }
};

TYPED_TEST_SUITE(NumericShuffleTensorInplaceFmaTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericShuffleTensorInplaceFmaTests,
           MatchesOutOfPlaceReferenceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const alpha = typename TestFixture::Accum{0.75};
    auto const beta = typename TestFixture::Accum{-1.25};

    auto const initial_a = TestFixture::make_tensor(1, basis);
    auto const b = TestFixture::make_tensor(2, basis);
    auto const c = TestFixture::make_tensor(3, basis);

    auto const actual = TestFixture::run_inplace_fma(
        basis,
        initial_a,
        b,
        c,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        alpha,
        beta);
    auto const expected = TestFixture::reference_fma(
        basis,
        initial_a,
        initial_a,
        b,
        c,
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        TestFixture::full_range(basis),
        alpha,
        beta);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericShuffleTensorInplaceFmaTests,
           MatchesOutOfPlaceReferenceForTruncatedViews) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    typename TestFixture::DegreeRange const a_range{1, 3};
    typename TestFixture::DegreeRange const b_range{1, 2};
    typename TestFixture::DegreeRange const c_range{1, TestFixture::depth};

    auto const initial_a = TestFixture::make_tensor(4, basis);
    auto const b = TestFixture::make_tensor(5, basis);
    auto const c = TestFixture::make_tensor(6, basis);

    auto const actual = TestFixture::run_inplace_fma(
        basis, initial_a, b, c, a_range, b_range, c_range);
    auto const expected = TestFixture::reference_fma(
        basis, initial_a, initial_a, b, c, a_range, a_range, b_range, c_range);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericShuffleTensorInplaceFmaTests,
           MatchesOutOfPlaceReferenceWithScaledAddendAndProduct) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const alpha = typename TestFixture::Accum{1.25};
    auto const beta = typename TestFixture::Accum{0.5};

    typename TestFixture::DegreeRange const a_range{1, TestFixture::depth};
    typename TestFixture::DegreeRange const b_range{0, 2};
    typename TestFixture::DegreeRange const c_range{1, TestFixture::depth};

    auto const initial_a = TestFixture::make_tensor(7, basis);
    auto const b = TestFixture::make_tensor(8, basis);
    auto const c = TestFixture::make_tensor(9, basis);

    auto const actual = TestFixture::run_inplace_fma(
        basis, initial_a, b, c, a_range, b_range, c_range, alpha, beta);
    auto const expected = TestFixture::reference_fma(
        basis,
        initial_a,
        initial_a,
        b,
        c,
        a_range,
        a_range,
        b_range,
        c_range,
        alpha,
        beta);
    TestFixture::expect_tensor_near(actual, expected);
}

} // namespace
