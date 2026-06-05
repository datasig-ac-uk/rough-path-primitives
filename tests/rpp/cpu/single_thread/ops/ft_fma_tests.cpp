#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/ft_fma.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_mul.hpp>
#include <rpp/cpu/single_thread/operations/linalg/vector_inplace_add.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_ft_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

using Helper = rpp::tests::PolynomialTensorHelper;
using Degree = Helper::Degree;
using Index = Helper::Index;
using Scalar = Helper::Scalar;

struct DegreeRange {
    Degree min;
    Degree max;
};

struct FmaViewCase {
    char const* name;
    DegreeRange out;
    DegreeRange a;
    DegreeRange b;
    DegreeRange c;
};

[[nodiscard]] bool contains(DegreeRange range, Degree degree) noexcept {
    return range.min <= degree && degree <= range.max;
}

class FreeTensorFmaTests : public testing::Test, public Helper {
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

    [[nodiscard]] static std::vector<Scalar>
    expected_fma(Basis const& basis,
                 std::vector<Scalar> const& initial_out,
                 std::vector<Scalar> const& a,
                 std::vector<Scalar> const& b,
                 std::vector<Scalar> const& c,
                 DegreeRange out_range,
                 DegreeRange a_range,
                 DegreeRange b_range,
                 DegreeRange c_range,
                 Scalar const& alpha = Scalar{1},
                 Scalar const& beta = Scalar{1}) {
        auto expected = initial_out;

        for_each_index(basis, [&](Degree degree, Index level_index) {
            if (!contains(out_range, degree)) {
                return;
            }

            Scalar entry{0};
            auto const global_index = static_cast<std::size_t>(
                basis.start_of_degree(degree) + level_index);

            if (contains(a_range, degree)) {
                entry += alpha * a[global_index];
            }

            auto const word = unpack_level_index(basis, degree, level_index);
            for (Degree mid = 0; mid <= degree; ++mid) {
                auto const split =
                    word.begin() + static_cast<std::ptrdiff_t>(mid);
                auto const lhs_index = pack_word(basis, word.begin(), split);
                auto const rhs_index = pack_word(basis, split, word.end());
                auto const rhs_degree = degree - mid;

                if (contains(b_range, mid) && contains(c_range, rhs_degree)) {
                    entry += beta *
                        b[static_cast<std::size_t>(basis.start_of_degree(mid) +
                                                   lhs_index)] *
                        c[static_cast<std::size_t>(
                            basis.start_of_degree(rhs_degree) + rhs_index)];
                }
            }

            expected[global_index] = entry;
        });

        return expected;
    }

    static void expect_fma_matches_reference(DegreeRange out_range,
                                             DegreeRange a_range,
                                             DegreeRange b_range,
                                             DegreeRange c_range,
                                             Scalar const& alpha = Scalar{1},
                                             Scalar const& beta = Scalar{1}) {
        auto const basis_data = BasisData(width, depth);
        auto const& basis = basis_data.basis;

        auto const initial_out = make_tensor('o', basis);
        auto out = initial_out;
        auto const a = make_tensor('a', basis);
        auto const b = make_tensor('b', basis);
        auto const c = make_tensor('c', basis);

        auto out_view = mutable_tensor_view(out, basis, out_range);
        auto const a_view = const_tensor_view(a, basis, a_range);
        auto const b_view = const_tensor_view(b, basis, b_range);
        auto const c_view = const_tensor_view(c, basis, c_range);

        auto const ctx = make_context();
        rpp::ops::FTFma<Strategy>{}(
            ctx, out_view, a_view, b_view, c_view, alpha, beta);

        EXPECT_EQ(out,
                  expected_fma(basis,
                               initial_out,
                               a,
                               b,
                               c,
                               out_range,
                               a_range,
                               b_range,
                               c_range,
                               alpha,
                               beta));
    }
};

TEST_F(FreeTensorFmaTests, AccumulatesConcatenationProduct) {
    expect_fma_matches_reference(
        {0, depth}, {0, depth}, {0, depth}, {0, depth});
}

TEST_F(FreeTensorFmaTests, HandlesTruncatedOperandDegreeRanges) {
    FmaViewCase const cases[] = {
        {"a has positive min degree",
         {0, depth},
         {2, depth},
         {0, depth},
         {0, depth}},
        {"a has truncated max degree",
         {0, depth},
         {0, 2},
         {0, depth},
         {0, depth}},
        {"b has positive min degree",
         {0, depth},
         {0, depth},
         {1, depth},
         {0, depth}},
        {"c has positive min degree",
         {0, depth},
         {0, depth},
         {0, depth},
         {1, depth}},
        {"b and c have truncated max degrees",
         {0, depth},
         {0, depth},
         {0, 2},
         {0, 1}},
        {"all operands are interior ranges",
         {0, depth},
         {1, 3},
         {1, 2},
         {1, 3}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_fma_matches_reference(
            test_case.out, test_case.a, test_case.b, test_case.c);
    }
}

TEST_F(FreeTensorFmaTests, RespectsTruncatedOutputDegreeRange) {
    FmaViewCase const cases[] = {
        {"output begins above zero",
         {2, depth},
         {0, depth},
         {0, depth},
         {0, depth}},
        {"output has truncated max degree",
         {0, 2},
         {0, depth},
         {0, depth},
         {0, depth}},
        {"output is an interior range", {1, 3}, {0, depth}, {0, 2}, {1, depth}},
    };

    for (auto const& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        expect_fma_matches_reference(
            test_case.out, test_case.a, test_case.b, test_case.c);
    }
}

TEST_F(FreeTensorFmaTests, HandlesScaledAddendAndProduct) {
    auto const alpha = make_scalar({{{{'p', 1}}, 2, 1}});
    auto const beta = make_scalar({{{{'q', 2}}, 3, 2}});

    expect_fma_matches_reference(
        {1, depth}, {1, 3}, {0, 2}, {1, depth}, alpha, beta);
}

TEST_F(FreeTensorFmaTests, MatchesMultiplyThenAddForTruncatedViews) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    DegreeRange const out_range{1, 3};
    DegreeRange const a_range{2, depth};
    DegreeRange const b_range{0, 2};
    DegreeRange const c_range{1, depth};
    auto const alpha = make_scalar({{{{'p', 3}}, 5, 3}});
    auto const beta = make_scalar({{{{'q', 4}}, 7, 5}});

    auto fma_out = std::vector<Scalar>(static_cast<std::size_t>(basis.size()));
    auto product_then_add =
        std::vector<Scalar>(static_cast<std::size_t>(basis.size()));
    auto const a = make_tensor('a', basis);
    auto const b = make_tensor('b', basis);
    auto const c = make_tensor('c', basis);

    auto fma_out_view = mutable_tensor_view(fma_out, basis, out_range);
    auto product_out_view =
        mutable_tensor_view(product_then_add, basis, out_range);
    auto const a_view = const_tensor_view(a, basis, a_range);
    auto const b_view = const_tensor_view(b, basis, b_range);
    auto const c_view = const_tensor_view(c, basis, c_range);

    auto const ctx = make_context();
    rpp::ops::FTFma<Strategy>{}(
        ctx, fma_out_view, a_view, b_view, c_view, alpha, beta);

    rpp::ops::FTMul<Strategy>{}(ctx, product_out_view, b_view, c_view, beta);
    rpp::ops::VectorInplaceAdd<Strategy>{}(
        ctx, product_out_view, a_view, alpha);

    EXPECT_EQ(fma_out, product_then_add);
}

TEST_F(FreeTensorFmaTests, KernelWrapperMatchesDirectOperation) {
    using Wrapper = rpp::tests::CpuKernelWrapperTestHelper;

    auto const basis_data = Wrapper::BasisData(Wrapper::width, Wrapper::depth);
    auto const& basis = basis_data.basis;
    auto const strategy = Wrapper::Strategy{};
    auto const alpha = Wrapper::make_scalar({{{{'p', 1}}, 3, 2}});
    auto const beta = Wrapper::make_scalar({{{{'q', 2}}, 5, 3}});

    auto actual = Wrapper::make_batch('o', basis);
    auto expected = actual;
    auto const a = Wrapper::make_batch('a', basis);
    auto const b = Wrapper::make_batch('b', basis);
    auto const c = Wrapper::make_batch('c', basis);


    const auto err = rpp::ops::ft_fma(strategy,
                                      {},
                                      Wrapper::tensor_batch(actual, basis),
                                      Wrapper::tensor_batch(a, basis),
                                      Wrapper::tensor_batch(b, basis),
                                      Wrapper::tensor_batch(c, basis),
                                      basis,
                                      Wrapper::tensor_count,
                                      alpha,
                                      beta);

    EXPECT_TRUE(static_cast<bool>(err)) << err.message();

    Wrapper::apply_direct<rpp::ops::FTFma<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto addend = Wrapper::tensor_view(a, basis, tensor_idx);
            auto left = Wrapper::tensor_view(b, basis, tensor_idx);
            auto right = Wrapper::tensor_view(c, basis, tensor_idx);
            op(ctx, out, addend, left, right, alpha, beta);
        });

    EXPECT_EQ(actual, expected);
}

template <typename Config>
class NumericFreeTensorFmaTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
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
    using Base::reference_mul;
    using Base::zero_tensor;

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_fma(Basis const& basis,
            std::vector<typename Base::Scalar> const& initial_out,
            std::vector<typename Base::Scalar> const& a,
            std::vector<typename Base::Scalar> const& b,
            std::vector<typename Base::Scalar> const& c,
            DegreeRange out_range,
            DegreeRange a_range,
            DegreeRange b_range,
            DegreeRange c_range,
            Accum alpha = Accum{1},
            Accum beta = Accum{1}) {
        auto out = initial_out;

        auto out_view = mutable_tensor_view(out, basis, out_range);
        auto const a_view = const_tensor_view(a, basis, a_range);
        auto const b_view = const_tensor_view(b, basis, b_range);
        auto const c_view = const_tensor_view(c, basis, c_range);

        auto const ctx = Base::make_context();
        rpp::ops::FTFma<Strategy>{}(
            ctx, out_view, a_view, b_view, c_view, alpha, beta);
        return out;
    }

    [[nodiscard]] static std::vector<typename Base::Scalar>
    run_mul(Basis const& basis,
            std::vector<typename Base::Scalar> const& initial_out,
            std::vector<typename Base::Scalar> const& lhs,
            std::vector<typename Base::Scalar> const& rhs,
            DegreeRange out_range,
            DegreeRange lhs_range,
            DegreeRange rhs_range,
            Accum beta = Accum{1}) {
        auto out = initial_out;
        auto out_view = mutable_tensor_view(out, basis, out_range);
        auto const lhs_view = const_tensor_view(lhs, basis, lhs_range);
        auto const rhs_view = const_tensor_view(rhs, basis, rhs_range);

        auto const ctx = Base::make_context();
        rpp::ops::FTMul<Strategy>{}(ctx, out_view, lhs_view, rhs_view, beta);
        return out;
    }

    static void run_vector_inplace_add(Basis const& basis,
                                       std::vector<typename Base::Scalar>& lhs,
                                       std::vector<typename Base::Scalar> const& rhs,
                                       DegreeRange lhs_range,
                                       DegreeRange rhs_range,
                                       Accum alpha = Accum{1}) {
        auto lhs_view = mutable_tensor_view(lhs, basis, lhs_range);
        auto const rhs_view = const_tensor_view(rhs, basis, rhs_range);

        auto const ctx = Base::make_context();
        rpp::ops::VectorInplaceAdd<Strategy>{}(ctx, lhs_view, rhs_view, alpha);
    }
};

TYPED_TEST_SUITE(NumericFreeTensorFmaTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericFreeTensorFmaTests, MatchesReferenceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const alpha = typename TestFixture::Accum{0.75};
    auto const beta = typename TestFixture::Accum{-1.25};

    auto const initial_out = TestFixture::make_tensor(1, basis);
    auto const a = TestFixture::make_tensor(2, basis);
    auto const b = TestFixture::make_tensor(3, basis);
    auto const c = TestFixture::make_tensor(4, basis);

    auto const actual =
        TestFixture::run_fma(basis,
                             initial_out,
                             a,
                             b,
                             c,
                             TestFixture::full_range(basis),
                             TestFixture::full_range(basis),
                             TestFixture::full_range(basis),
                             TestFixture::full_range(basis),
                             alpha,
                             beta);
    auto const expected =
        TestFixture::reference_fma(basis,
                                   initial_out,
                                   a,
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

TYPED_TEST(NumericFreeTensorFmaTests, MatchesMultiplyThenAddReference) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    typename TestFixture::DegreeRange const out_range{1, 3};
    typename TestFixture::DegreeRange const a_range{2, TestFixture::depth};
    typename TestFixture::DegreeRange const b_range{0, 2};
    typename TestFixture::DegreeRange const c_range{1, TestFixture::depth};
    auto const alpha = typename TestFixture::Accum{1.25};
    auto const beta = typename TestFixture::Accum{0.5};

    auto const initial_out = TestFixture::zero_tensor(basis);
    auto const a = TestFixture::make_tensor(5, basis);
    auto const b = TestFixture::make_tensor(6, basis);
    auto const c = TestFixture::make_tensor(7, basis);

    auto const actual = TestFixture::run_fma(
        basis, initial_out, a, b, c, out_range, a_range, b_range, c_range, alpha, beta);

    auto expected = TestFixture::run_mul(
        basis, initial_out, b, c, out_range, b_range, c_range, beta);
    TestFixture::run_vector_inplace_add(
        basis, expected, a, out_range, a_range, alpha);

    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericFreeTensorFmaTests, RespectsTruncatedOperandAndOutputViews) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;

    typename TestFixture::DegreeRange const out_range{1, 3};
    typename TestFixture::DegreeRange const a_range{1, 3};
    typename TestFixture::DegreeRange const b_range{1, 2};
    typename TestFixture::DegreeRange const c_range{1, TestFixture::depth};

    auto const initial_out = TestFixture::make_tensor(8, basis);
    auto const a = TestFixture::make_tensor(9, basis);
    auto const b = TestFixture::make_tensor(10, basis);
    auto const c = TestFixture::make_tensor(11, basis);

    auto const actual = TestFixture::run_fma(
        basis, initial_out, a, b, c, out_range, a_range, b_range, c_range);
    auto const expected = TestFixture::reference_fma(
        basis, initial_out, a, b, c, out_range, a_range, b_range, c_range);
    TestFixture::expect_tensor_near(actual, expected);
}

} // namespace
