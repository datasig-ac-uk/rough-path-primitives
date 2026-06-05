#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/st_fma.hpp>
#include <rpp/views/views.hpp>

#include "cpu_kernel_wrapper_test_helper.hpp"
#include "cpu_typed_st_ops_test_helper.hpp"
#include "polynomial_tensor_helper.hpp"

namespace {

class ShuffleTensorFmaTests : public testing::Test,
                              public rpp::tests::PolynomialTensorHelper {
protected:
    static constexpr Degree width = 3;
    static constexpr Degree depth = 4;

    [[nodiscard]] static std::vector<Scalar> zero_outside_range(
        std::vector<Scalar> value,
        Basis const& basis,
        Degree min_degree,
        Degree max_degree) {
        for (Degree degree = 0; degree <= basis.depth; ++degree) {
            if (degree < min_degree || degree > max_degree) {
                const auto begin = basis.start_of_degree(degree);
                const auto end = basis.end_of_degree(degree);
                std::fill(value.begin() + begin, value.begin() + end, Scalar{0});
            }
        }
        return value;
    }
};

TEST_F(ShuffleTensorFmaTests, AccumulatesShuffleProduct) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('c', basis);
    auto const lhs = make_tensor('a', basis);
    auto const rhs = make_tensor('b', basis);

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> lhs_view(lhs.data(), basis);
    TensorView<Scalar const*> rhs_view(rhs.data(), basis);

    auto const ctx = make_context();
    rpp::ops::STFma<Strategy>{}(ctx, out_view, out_view, lhs_view, rhs_view);

    std::vector<Scalar> expected;
    expected.reserve(static_cast<std::size_t>(basis.size()));

    for_each_index(
        basis, [&expected, &basis](Degree degree, Index level_index) {
            auto const word = unpack_level_index(basis, degree, level_index);
            auto entry = symbol('c', basis, degree, level_index);
            auto const mask_count = std::uint32_t{1} << degree;

            for (std::uint32_t mask = 0; mask < mask_count; ++mask) {
                std::vector<std::size_t> lhs_word;
                std::vector<std::size_t> rhs_word;
                lhs_word.reserve(word.size());
                rhs_word.reserve(word.size());

                for (Degree i = 0; i < degree; ++i) {
                    if (((mask >> i) & std::uint32_t{1}) != 0) {
                        lhs_word.push_back(word[static_cast<std::size_t>(i)]);
                    }
                    else {
                        rhs_word.push_back(word[static_cast<std::size_t>(i)]);
                    }
                }

                auto const lhs_degree = static_cast<Degree>(lhs_word.size());
                auto const rhs_degree = static_cast<Degree>(rhs_word.size());
                auto const lhs_index =
                    pack_word(basis, lhs_word.begin(), lhs_word.end());
                auto const rhs_index =
                    pack_word(basis, rhs_word.begin(), rhs_word.end());

                entry += make_scalar(
                    {{{{'a', symbol_index(basis, lhs_degree, lhs_index)},
                       {'b', symbol_index(basis, rhs_degree, rhs_index)}},
                      1,
                      1}});
            }

            expected.emplace_back(std::move(entry));
        });

    EXPECT_EQ(out, expected);
}

TEST_F(ShuffleTensorFmaTests, KernelWrapperMatchesDirectOperation) {
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

    auto const err =
        rpp::ops::st_fma(strategy,
                         typename Wrapper::Strategy::LaunchConfig{},
                         Wrapper::tensor_batch(actual, basis),
                         Wrapper::tensor_batch(a, basis),
                         Wrapper::tensor_batch(b, basis),
                         Wrapper::tensor_batch(c, basis),
                         basis,
                         Wrapper::tensor_count,
                         alpha,
                         beta);
    EXPECT_TRUE(static_cast<bool>(err)) << err.message();
    Wrapper::apply_direct<rpp::ops::STFma<Wrapper::Strategy>>(
        basis, [&](auto const& op, auto const& ctx, Wrapper::Index tensor_idx) {
            auto out = Wrapper::tensor_view(expected, basis, tensor_idx);
            auto addend = Wrapper::tensor_view(a, basis, tensor_idx);
            auto left = Wrapper::tensor_view(b, basis, tensor_idx);
            auto right = Wrapper::tensor_view(c, basis, tensor_idx);
            op(ctx, out, addend, left, right, alpha, beta);
        });

    EXPECT_EQ(actual, expected);
}

TEST_F(ShuffleTensorFmaTests, MatchesZeroExtendedOperandsForTruncatedViews) {
    auto const basis_data = BasisData(width, depth);
    auto const& basis = basis_data.basis;

    auto out = make_tensor('o', basis);
    auto const addend = make_tensor('a', basis);
    auto const lhs = make_tensor('b', basis);
    auto const rhs = make_tensor('c', basis);
    auto expected = out;

    auto const addend_min = Degree{1};
    auto const addend_max = Degree{3};
    auto const lhs_min = Degree{0};
    auto const lhs_max = Degree{2};
    auto const rhs_min = Degree{2};
    auto const rhs_max = Degree{4};
    auto const alpha = make_scalar({{{{'p', 1}}, 3, 2}});
    auto const beta = make_scalar({{{{'q', 2}}, 5, 3}});

    TensorView<Scalar*> out_view(out.data(), basis);
    TensorView<Scalar const*> addend_view(
        addend.data(), basis, addend_min, addend_max);
    TensorView<Scalar const*> lhs_view(lhs.data(), basis, lhs_min, lhs_max);
    TensorView<Scalar const*> rhs_view(rhs.data(), basis, rhs_min, rhs_max);

    auto const ctx = make_context();
    rpp::ops::STFma<Strategy>{}(
        ctx, out_view, addend_view, lhs_view, rhs_view, alpha, beta);

    auto const extended_addend =
        zero_outside_range(addend, basis, addend_min, addend_max);
    auto const extended_lhs = zero_outside_range(lhs, basis, lhs_min, lhs_max);
    auto const extended_rhs = zero_outside_range(rhs, basis, rhs_min, rhs_max);
    TensorView<Scalar*> expected_view(expected.data(), basis);
    TensorView<Scalar const*> expected_addend_view(extended_addend.data(), basis);
    TensorView<Scalar const*> expected_lhs_view(extended_lhs.data(), basis);
    TensorView<Scalar const*> expected_rhs_view(extended_rhs.data(), basis);
    rpp::ops::STFma<Strategy>{}(ctx,
                                expected_view,
                                expected_addend_view,
                                expected_lhs_view,
                                expected_rhs_view,
                                alpha,
                                beta);

    EXPECT_EQ(out, expected);
}

template <typename Config>
class NumericShuffleTensorFmaTests
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
        rpp::ops::STFma<Strategy>{}(
            ctx, out_view, a_view, b_view, c_view, alpha, beta);
        return out;
    }
};

TYPED_TEST_SUITE(NumericShuffleTensorFmaTests,
                 rpp::tests::TypedCpuFreeTensorTestTypes);

TYPED_TEST(NumericShuffleTensorFmaTests, MatchesReferenceOnFullView) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const alpha = typename TestFixture::Accum{0.75};
    auto const beta = typename TestFixture::Accum{-1.25};

    auto const initial_out = TestFixture::make_tensor(1, basis);
    auto const a = TestFixture::make_tensor(2, basis);
    auto const b = TestFixture::make_tensor(3, basis);
    auto const c = TestFixture::make_tensor(4, basis);

    auto const actual = TestFixture::run_fma(
        basis,
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
    auto const expected = TestFixture::reference_fma(
        basis,
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

TYPED_TEST(NumericShuffleTensorFmaTests, MatchesMultiplyThenAddReference) {
    auto const basis_data =
        typename TestFixture::BasisData(TestFixture::width, TestFixture::depth);
    auto const& basis = basis_data.basis;
    auto const alpha = typename TestFixture::Accum{1.25};
    auto const beta = typename TestFixture::Accum{0.5};

    typename TestFixture::DegreeRange const out_range{1, 3};
    typename TestFixture::DegreeRange const a_range{1, 3};
    typename TestFixture::DegreeRange const b_range{0, 2};
    typename TestFixture::DegreeRange const c_range{2, TestFixture::depth};

    auto const initial_out = TestFixture::zero_tensor(basis);
    auto const a = TestFixture::make_tensor(5, basis);
    auto const b = TestFixture::make_tensor(6, basis);
    auto const c = TestFixture::make_tensor(7, basis);

    auto const actual = TestFixture::run_fma(
        basis, initial_out, a, b, c, out_range, a_range, b_range, c_range, alpha, beta);
    auto const expected = TestFixture::reference_fma(
        basis, initial_out, a, b, c, out_range, a_range, b_range, c_range, alpha, beta);
    TestFixture::expect_tensor_near(actual, expected);
}

TYPED_TEST(NumericShuffleTensorFmaTests,
           RespectsTruncatedOperandAndOutputViews) {
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
