#include <algorithm>
#include <array>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/cpu/single_thread/operations/basic/detail/shuffle_adjoint_op_loop.hpp>
#include <rpp/operations/implementation/word_half_shuffle_product.hpp>
#include <rpp/operations/implementation/word_shuffle_adjoint_multiply.hpp>

#include "cpu_typed_ft_ops_test_helper.hpp"

namespace {

using Basis = rpp::tests::CpuTypedTensorBasis;
using Degree = Basis::Degree;
using Index = Basis::Index;
using Letter = rpp::tests::CpuTypedNumericTestArchitecture::Letter;

Index pack_word(std::initializer_list<Index> letters, Index width) {
    Index result{0};
    for (auto letter : letters) {
        result = result * width + letter;
    }
    return result;
}

// The helper expects each source word in least-significant-letter-first order,
// with the operator word immediately after the output word.
auto make_letters(Index out_index,
                  Degree out_degree,
                  Index op_index,
                  Degree op_degree,
                  Index width) {
    std::array<Letter,
               rpp::tests::PolynomialTensorHelper::TestArchitecture::max_depth>
        letters{};
    for (Degree pos = 0; pos < out_degree; ++pos) {
        letters[pos] = static_cast<Letter>(out_index % width);
        out_index /= width;
    }
    for (Degree pos = 0; pos < op_degree; ++pos) {
        letters[out_degree + pos] = static_cast<Letter>(op_index % width);
        op_index /= width;
    }
    return letters;
}

// Independent reference: recursively take the next letter of either source.
// No basis packing helpers or bitmasks are used. Duplicate words are retained.
auto reference_interleavings(Index out_index,
                             Degree out_degree,
                             Index op_index,
                             Degree op_degree,
                             Index width) {
    auto decode = [width](Index idx, Degree degree) {
        std::vector<Index> word(static_cast<std::size_t>(degree));
        for (Degree pos = degree; pos > 0; --pos) {
            word[pos - 1] = idx % width;
            idx /= width;
        }
        return word;
    };
    auto const out = decode(out_index, out_degree);
    auto const op = decode(op_index, op_degree);
    std::vector<Index> result;
    std::function<void(Degree, Degree, Index)> append =
        [&](Degree out_pos, Degree op_pos, Index idx) {
            if (out_pos == out_degree && op_pos == op_degree) {
                result.push_back(idx);
                return;
            }
            if (out_pos < out_degree) {
                append(out_pos + 1, op_pos, idx * width + out[out_pos]);
            }
            if (op_pos < op_degree) {
                append(out_pos, op_pos + 1, idx * width + op[op_pos]);
            }
        };
    append(0, 0, 0);
    return result;
}

template <typename Config>
class WordShuffleAdjointMultiplyTests
    : public rpp::tests::TypedCpuFreeTensorOpTestBase<Config> {
protected:
    using Base = rpp::tests::TypedCpuFreeTensorOpTestBase<Config>;
    using typename Base::Accum;
    using typename Base::ConstTensorView;
    using typename Base::Scalar;

    struct RecordingGetter {
        std::vector<Index>& indices;
        Degree expected_degree;

        Scalar operator()(ConstTensorView const& tensor,
                          Degree degree,
                          Index idx) const {
            EXPECT_EQ(degree, expected_degree);
            indices.push_back(idx);
            return tensor.degree_view(degree)[idx];
        }
    };

    static auto make_values(Basis const& basis, unsigned seed) {
        auto values = Base::zero_tensor(basis);
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<Scalar>(
                static_cast<float>((i * 7 + seed) % 17 + 1) / 8.0f);
        }
        return values;
    }

    static void check_word_pair(Basis const& basis,
                                Index out_idx,
                                Degree out_degree,
                                Index op_idx,
                                Degree op_degree) {
        SCOPED_TRACE(testing::Message()
                     << "out=" << out_degree << ":" << out_idx
                     << " op=" << op_degree << ":" << op_idx);
        auto const arg = make_values(basis, 3);
        auto const op = make_values(basis, 5);
        ConstTensorView arg_view(arg.data(), basis), op_view(op.data(), basis);
        auto expected_indices = reference_interleavings(
            out_idx, out_degree, op_idx, op_degree, basis.width);
        Accum expected{0};
        for (auto idx : expected_indices) {
            expected +=
                static_cast<Accum>(
                    op[basis.start_of_degree(op_degree) + op_idx]) *
                static_cast<Accum>(
                    arg[basis.start_of_degree(out_degree + op_degree) + idx]);
        }
        std::vector<Index> actual_indices;
        RecordingGetter getter{actual_indices, out_degree + op_degree};
        auto letters =
            make_letters(out_idx, out_degree, op_idx, op_degree, basis.width);
        rpp::Span<Letter> letter_span{
            letters.data(), static_cast<std::size_t>(out_degree + op_degree)};
        auto const actual = rpp::ops::common::word_shuffle_adjoint_multiply(
            Base::make_context(),
            out_idx,
            out_degree,
            op_idx,
            op_degree,
            letter_span,
            arg_view,
            op_view,
            getter);
        std::sort(actual_indices.begin(), actual_indices.end());
        std::sort(expected_indices.begin(), expected_indices.end());
        EXPECT_EQ(actual_indices, expected_indices);
        EXPECT_EQ(actual, expected);
    }
};

using WordShuffleTypes =
    testing::Types<rpp::tests::TypedScalarAccumConfig<float, float>,
                   rpp::tests::TypedScalarAccumConfig<float, double>,
                   rpp::tests::TypedScalarAccumConfig<double, double>>;

TYPED_TEST_SUITE(WordShuffleAdjointMultiplyTests, WordShuffleTypes);

TYPED_TEST(WordShuffleAdjointMultiplyTests,
           DistinctLettersPreserveBothSourceOrders) {
    auto const basis_data = typename TestFixture::BasisData(4, 4);
    auto const& basis = basis_data.basis;
    auto const arg = TestFixture::make_values(basis, 3);
    auto const op = TestFixture::make_values(basis, 5);
    typename TestFixture::ConstTensorView arg_view(arg.data(), basis),
        op_view(op.data(), basis);
    std::vector<Index> actual_indices;
    typename TestFixture::RecordingGetter getter{actual_indices, 4};
    auto letters = make_letters(
        pack_word({2, 3}, 4), 2, pack_word({0, 1}, 4), 2, basis.width);
    rpp::Span<Letter, 4> letter_span{letters.data(), 4};
    rpp::ops::common::word_shuffle_adjoint_multiply(TestFixture::make_context(),
                                                    pack_word({2, 3}, 4),
                                                    Degree{2},
                                                    pack_word({0, 1}, 4),
                                                    Degree{2},
                                                    letter_span,
                                                    arg_view,
                                                    op_view,
                                                    getter);
    std::vector<Index> expected_indices{pack_word({0, 1, 2, 3}, 4),
                                        pack_word({0, 2, 1, 3}, 4),
                                        pack_word({0, 2, 3, 1}, 4),
                                        pack_word({2, 0, 1, 3}, 4),
                                        pack_word({2, 0, 3, 1}, 4),
                                        pack_word({2, 3, 0, 1}, 4)};
    std::sort(actual_indices.begin(), actual_indices.end());
    std::sort(expected_indices.begin(), expected_indices.end());
    EXPECT_EQ(actual_indices, expected_indices);
    // Single-source words expose reversal without depending on mask order.
    TestFixture::check_word_pair(basis, 0, 0, pack_word({0, 1, 2, 3}, 4), 4);
    TestFixture::check_word_pair(basis, pack_word({0, 1, 2, 3}, 4), 4, 0, 0);
}

TYPED_TEST(WordShuffleAdjointMultiplyTests,
           EverySmallWordPairMatchesRecursiveInterleavings) {
    for (Degree width : {1, 2, 3}) {
        SCOPED_TRACE(width);
        auto const basis_data = typename TestFixture::BasisData(width, 4);
        auto const& basis = basis_data.basis;
        for (Degree out_degree = 0; out_degree <= basis.depth; ++out_degree) {
            for (Degree op_degree = 0; out_degree + op_degree <= basis.depth;
                 ++op_degree) {
                for (Index out_idx = 0;
                     out_idx < basis.size_of_degree(out_degree);
                     ++out_idx) {
                    for (Index op_idx = 0;
                         op_idx < basis.size_of_degree(op_degree);
                         ++op_idx) {
                        TestFixture::check_word_pair(
                            basis, out_idx, out_degree, op_idx, op_degree);
                    }
                }
            }
        }
    }
}

TYPED_TEST(WordShuffleAdjointMultiplyTests,
           RepeatedLettersRetainMultiplicityAtMaximumDepth) {
    auto const basis_data = typename TestFixture::BasisData(1, 16);
    auto const& basis = basis_data.basis;
    TestFixture::check_word_pair(basis, 0, 8, 0, 8);
    TestFixture::check_word_pair(basis, 0, 0, 0, 16);
    TestFixture::check_word_pair(basis, 0, 16, 0, 0);
}

TYPED_TEST(WordShuffleAdjointMultiplyTests,
           PrefixGettersShiftOnlyTheRequestedInputs) {
    using Scalar = typename TestFixture::Scalar;
    using Accum = typename TestFixture::Accum;
    using View = typename TestFixture::ConstTensorView;
    auto const basis_data = typename TestFixture::BasisData(3, 4);
    auto const& basis = basis_data.basis;
    auto const arg = TestFixture::make_values(basis, 3);
    auto const op = TestFixture::make_values(basis, 5);
    View arg_view(arg.data(), basis), op_view(op.data(), basis);
    for (Index letter = 0; letter < basis.width; ++letter) {
        for (Degree out_degree = 0; out_degree <= 2; ++out_degree) {
            for (Degree op_degree = 0; op_degree <= 2; ++op_degree) {
                auto const degree = out_degree + op_degree;
                if (degree + 1 > basis.depth) {
                    continue;
                }
                for (Index out_idx = 0;
                     out_idx < basis.size_of_degree(out_degree);
                     ++out_idx) {
                    for (Index op_idx = 0;
                         op_idx < basis.size_of_degree(op_degree);
                         ++op_idx) {
                        SCOPED_TRACE(testing::Message()
                                     << "letter=" << letter
                                     << " out=" << out_degree << ":" << out_idx
                                     << " op=" << op_degree << ":" << op_idx);
                        auto const indices =
                            reference_interleavings(out_idx,
                                                    out_degree,
                                                    op_idx,
                                                    op_degree,
                                                    basis.width);
                        Accum expected_arg_only{0}, expected_both{0};
                        for (auto idx : indices) {
                            auto const arg_value = static_cast<Accum>(
                                arg[basis.start_of_degree(degree + 1) +
                                    letter * basis.size_of_degree(degree) +
                                    idx]);
                            expected_arg_only +=
                                static_cast<Accum>(
                                    op[basis.start_of_degree(op_degree) +
                                       op_idx]) *
                                arg_value;
                            expected_both +=
                                static_cast<Accum>(
                                    op[basis.start_of_degree(op_degree + 1) +
                                       letter *
                                           basis.size_of_degree(op_degree) +
                                       op_idx]) *
                                arg_value;
                        }
                        auto letters = make_letters(out_idx,
                                                    out_degree,
                                                    op_idx,
                                                    op_degree,
                                                    basis.width);
                        rpp::Span<Letter> letter_span{
                            letters.data(), static_cast<std::size_t>(degree)};
                        auto const actual_arg_only =
                            rpp::ops::common::word_shuffle_adjoint_multiply(
                                TestFixture::make_context(),
                                out_idx,
                                out_degree,
                                op_idx,
                                op_degree,
                                letter_span,
                                arg_view,
                                op_view,
                                rpp::ops::common::detail::PrefixLetterGetter<
                                    View>{letter});
                        auto const actual_both =
                            rpp::ops::common::word_shuffle_adjoint_multiply(
                                TestFixture::make_context(),
                                out_idx,
                                out_degree,
                                op_idx,
                                op_degree,
                                letter_span,
                                arg_view,
                                op_view,
                                rpp::ops::common::detail::PrefixLetterGetter<
                                    View>{letter},
                                rpp::ops::common::detail::PrefixLetterGetter<
                                    View>{letter});
                        EXPECT_EQ(actual_arg_only, expected_arg_only);
                        EXPECT_EQ(actual_both, expected_both);
                    }
                }
            }
        }
    }
}

TYPED_TEST(WordShuffleAdjointMultiplyTests, OpLoopSupportsBothPrefixPatterns) {
    using Accum = typename TestFixture::Accum;
    using View = typename TestFixture::ConstTensorView;
    auto const basis_data = typename TestFixture::BasisData(3, 4);
    auto const& basis = basis_data.basis;
    auto const arg = TestFixture::make_values(basis, 3);
    auto const op = TestFixture::make_values(basis, 5);
    View arg_view(arg.data(), basis), op_view(op.data(), basis);
    for (Index letter = 0; letter < basis.width; ++letter) {
        for (Degree out_degree = 0; out_degree <= 1; ++out_degree) {
            for (Index out_index = 0;
                 out_index < basis.size_of_degree(out_degree);
                 ++out_index) {
                // Singleton bounds exercise both endpoints, including zero;
                // the wider range checks accumulation across degrees.
                for (auto bounds : {std::array<Degree, 2>{0, 0},
                                    std::array<Degree, 2>{2, 2},
                                    std::array<Degree, 2>{0, 2},
                                    std::array<Degree, 2>{2, 1}}) {
                    SCOPED_TRACE(testing::Message()
                                 << "letter=" << letter << " out=" << out_degree
                                 << ":" << out_index << " bounds=" << bounds[0]
                                 << ":" << bounds[1]);
                    Accum expected_arg_only{0}, expected_both{0};
                    for (Degree q = bounds[0]; q <= bounds[1]; ++q) {
                        for (Index u = 0; u < basis.size_of_degree(q); ++u) {
                            for (auto idx :
                                 reference_interleavings(out_index,
                                                         out_degree,
                                                         u,
                                                         q,
                                                         basis.width)) {
                                auto const arg_value = static_cast<Accum>(
                                    arg[basis.start_of_degree(out_degree + q +
                                                              1) +
                                        letter *
                                            basis.size_of_degree(out_degree +
                                                                 q) +
                                        idx]);
                                expected_arg_only +=
                                    static_cast<Accum>(
                                        op[basis.start_of_degree(q) + u]) *
                                    arg_value;
                                expected_both +=
                                    static_cast<Accum>(
                                        op[basis.start_of_degree(q + 1) +
                                           letter * basis.size_of_degree(q) +
                                           u]) *
                                    arg_value;
                            }
                        }
                    }
                    auto letters =
                        make_letters(out_index, out_degree, 0, 0, basis.width);
                    auto const original_letters = letters;
                    rpp::Span<Letter> letter_span{letters.data(),
                                                  letters.size()};
                    auto prefix =
                        rpp::ops::common::detail::PrefixLetterGetter<View>{
                            letter};
                    auto const actual_arg_only =
                        rpp::ops::common::shuffle_adjoint_op_loop(
                            TestFixture::make_context(),
                            out_index,
                            out_degree,
                            letter_span,
                            op_view,
                            arg_view,
                            bounds[0],
                            bounds[1],
                            prefix);
                    auto const actual_both =
                        rpp::ops::common::shuffle_adjoint_op_loop(
                            TestFixture::make_context(),
                            out_index,
                            out_degree,
                            letter_span,
                            op_view,
                            arg_view,
                            bounds[0],
                            bounds[1],
                            prefix,
                            prefix);
                    EXPECT_EQ(actual_arg_only, expected_arg_only);
                    EXPECT_EQ(actual_both, expected_both);
                    EXPECT_TRUE(std::equal(letters.begin(),
                                           letters.begin() + out_degree,
                                           original_letters.begin()));
                }
            }
        }
    }
}

// Genuine noncommuting coefficients distinguish the default and reversed
// actions.
struct Matrix2 {
    std::array<int, 4> values{};
    Matrix2(int scalar = 0) : values{scalar, 0, 0, scalar} {}
    Matrix2(int a, int b, int c, int d) : values{a, b, c, d} {}
    Matrix2& operator+=(Matrix2 const& rhs) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] += rhs.values[i];
        }
        return *this;
    }
    friend Matrix2 operator*(Matrix2 const& lhs, Matrix2 const& rhs) {
        auto const& a = lhs.values;
        auto const& b = rhs.values;
        return {a[0] * b[0] + a[1] * b[2],
                a[0] * b[1] + a[1] * b[3],
                a[2] * b[0] + a[3] * b[2],
                a[2] * b[1] + a[3] * b[3]};
    }
};

TEST(WordShuffleAdjointMultiplyCoefficientTests,
     CustomMultiplyPreservesNoncommutativeOperandOrder) {
    using Helper = rpp::tests::PolynomialTensorHelper;
    using Strategy =
        rpp::cpu::strategies::SingleThreadStrategy<Matrix2,
                                                   Helper::TestArchitecture>;
    using View = rpp::DenseTensorView<Matrix2 const*, Basis>;
    auto const basis_data = Helper::BasisData(2, 2);
    auto const& basis = basis_data.basis;
    Matrix2 const op_value{1, 1, 0, 1}, arg_value{1, 0, 1, 1};
    std::vector<Matrix2> op(basis.size(), op_value),
        arg(basis.size(), arg_value);
    View op_view(op.data(), basis), arg_view(arg.data(), basis);
    auto reverse_multiply = [](Matrix2 const& lhs, Matrix2 const& rhs) {
        return rhs * lhs;
    };
    auto const ctx = Strategy::make_context(nullptr);
    for (Degree op_degree : {0, 1}) {
        Matrix2 expected = op_value * arg_value;
        Matrix2 expected_reverse = arg_value * op_value;
        if (op_degree == 1) {
            expected += op_value * arg_value;
            expected_reverse += arg_value * op_value;
        }
        ASSERT_NE(expected.values, expected_reverse.values);
        auto letters = make_letters(0, 1, 0, op_degree, basis.width);
        rpp::Span<Letter> letter_span{letters.data(),
                                      static_cast<std::size_t>(1 + op_degree)};
        auto const actual =
            rpp::ops::common::word_shuffle_adjoint_multiply(ctx,
                                                            Index{0},
                                                            Degree{1},
                                                            Index{0},
                                                            op_degree,
                                                            letter_span,
                                                            arg_view,
                                                            op_view);
        auto const actual_reverse =
            rpp::ops::common::word_shuffle_adjoint_multiply(
                ctx,
                Index{0},
                Degree{1},
                Index{0},
                op_degree,
                letter_span,
                arg_view,
                op_view,
                rpp::ops::common::detail::DefaultGetter<View>{},
                rpp::ops::common::detail::DefaultGetter<View>{},
                reverse_multiply);
        EXPECT_EQ(actual.values, expected.values);
        EXPECT_EQ(actual_reverse.values, expected_reverse.values);
    }
}

} // namespace
