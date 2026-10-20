#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <rpp/basis/hall_basis.hpp>
#include <rpp/basis/lie_multiplication.hpp>
#include <rpp/sparse/l2t_matrix.hpp>
#include <rpp/sparse/t2l_matrix.hpp>

namespace {

using Architecture = rpp::arch::NativeArchitecture;
using Index = Architecture::Index;
using Expansion = std::map<std::string, double>;

// Compute in the free associative algebra, independently of the matrix
// constructors and the Lie multiplication cache.
Expansion commutator(Expansion const& left, Expansion const& right) {
    Expansion result;
    for (auto const& [u, a] : left) {
        for (auto const& [v, b] : right) {
            result[u + v] += a * b;
            result[v + u] -= b * a;
        }
    }
    return result;
}

std::vector<Expansion> expand_hall(rpp::basis::StandardLieBasis const& basis) {
    std::vector<Expansion> result(basis.true_size());
    for (Index key = 1; key < basis.true_size(); ++key) {
        auto const [left, right] = basis[key];
        if (left == 0) {
            result[key] = {{std::string(1, static_cast<char>('a' + right - 1)), 1}};
        }
        else {
            result[key] = commutator(result[left], result[right]);
        }
    }
    return result;
}

std::vector<std::string> tensor_words(int width, int depth) {
    std::vector<std::string> result{std::string{}};
    std::size_t begin = 0;
    std::size_t end = 1;
    for (int degree = 1; degree <= depth; ++degree) {
        for (auto i = begin; i < end; ++i) {
            for (int letter = 0; letter < width; ++letter) {
                result.push_back(result[i] + static_cast<char>('a' + letter));
            }
        }
        begin = end;
        end = result.size();
    }
    return result;
}

Expansion right_bracket(std::string const& word) {
    if (word.empty()) {
        return {};
    }
    Expansion result{{word.substr(word.size() - 1), 1}};
    for (auto i = word.size() - 1; i > 0; --i) {
        result = commutator(Expansion{{word.substr(i - 1, 1), 1}}, result);
    }
    return result;
}

template <typename Matrix>
void expect_canonical_csc(Matrix const& matrix, Index rows, Index cols) {
    ASSERT_EQ(matrix.view().rows(), rows);
    ASSERT_EQ(matrix.view().cols(), cols);
    ASSERT_EQ(matrix.offsets()[0], 0);
    for (Index col = 0; col < cols; ++col) {
        SCOPED_TRACE(::testing::Message() << "column " << col);
        const auto begin = matrix.offsets()[col];
        const auto end = matrix.offsets()[col + 1];
        ASSERT_GE(begin, 0);
        ASSERT_LE(begin, end);
        ASSERT_LE(end, matrix.nnz());
        for (auto i = begin; i < end; ++i) {
            ASSERT_GE(matrix.indices()[i], 0);
            ASSERT_LT(matrix.indices()[i], rows);
            EXPECT_NE(matrix.data()[i], 0);
            if (i > begin) {
                EXPECT_LT(matrix.indices()[i - 1], matrix.indices()[i]);
            }
        }
    }
    EXPECT_EQ(matrix.offsets()[cols], matrix.nnz());
}

template <typename Scalar>
class LieTensorMatrixTests : public ::testing::Test {};

using Scalars = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LieTensorMatrixTests, Scalars);

TYPED_TEST(LieTensorMatrixTests, EveryL2TColumnMatchesItsHallTree) {
    auto const hall = rpp::basis::HallBasis<Architecture>(3, 4);
    auto const basis = hall.to_lie_basis();
    auto const words = tensor_words(3, 4);
    auto const expansions = expand_hall(basis);
    auto const matrix = rpp::sparse::make_l2t_matrix<TypeParam>(basis);
    ASSERT_NO_FATAL_FAILURE(expect_canonical_csc(matrix, 121, 32));

    for (Index col = 0; col < 32; ++col) {
        SCOPED_TRACE(::testing::Message() << "Lie key " << col + 1);
        auto difference = expansions[col + 1];
        for (auto i = matrix.offsets()[col]; i < matrix.offsets()[col + 1]; ++i) {
            difference[words[matrix.indices()[i]]] -= matrix.data()[i];
        }
        for (auto const& [word, value] : difference) {
            EXPECT_EQ(value, 0) << "tensor word " << word;
        }
    }
}

TYPED_TEST(LieTensorMatrixTests, EveryT2LColumnMatchesNormalizedRightBracketing) {
    auto const hall = rpp::basis::HallBasis<Architecture>(3, 4);
    auto const basis = hall.to_lie_basis();
    auto const words = tensor_words(3, 4);
    auto const expansions = expand_hall(basis);
    rpp::basis::LieMultiplicationCache<Architecture> cache(basis);
    auto const matrix = rpp::sparse::make_t2l_matrix<TypeParam>(basis, cache);
    ASSERT_NO_FATAL_FAILURE(expect_canonical_csc(matrix, 32, 121));
    const auto tolerance = 32 * std::numeric_limits<TypeParam>::epsilon();

    for (Index col = 0; col < 121; ++col) {
        SCOPED_TRACE(::testing::Message() << "tensor word " << words[col]);
        auto difference = right_bracket(words[col]);
        for (auto& [word, value] : difference) {
            value /= static_cast<double>(words[col].size());
        }
        for (auto i = matrix.offsets()[col]; i < matrix.offsets()[col + 1]; ++i) {
            const auto key = basis.idx_to_key(matrix.indices()[i]);
            EXPECT_EQ(basis.degree(key), words[col].size());
            for (auto const& [word, value] : expansions[key]) {
                difference[word] -= matrix.data()[i] * value;
            }
        }
        for (auto const& [word, value] : difference) {
            EXPECT_NEAR(value, 0, tolerance) << "expanded word " << word;
        }
    }
}

TYPED_TEST(LieTensorMatrixTests, LieToTensorToLieIsIdentity) {
    const auto tolerance = 32 * std::numeric_limits<TypeParam>::epsilon();
    for (int width : {2, 3}) {
        for (int depth : {0, 1, 2, 4}) {
            SCOPED_TRACE(::testing::Message() << "width " << width
                                             << ", depth " << depth);
            auto const hall = rpp::basis::HallBasis<Architecture>(width, depth);
            auto const basis = hall.to_lie_basis();
            const auto lie_size = basis.size();
            const auto tensor_size = static_cast<Index>(tensor_words(width, depth).size());
            rpp::basis::LieMultiplicationCache<Architecture> cache(basis);
            auto const l2t = rpp::sparse::make_l2t_matrix<TypeParam>(basis);
            auto const t2l = rpp::sparse::make_t2l_matrix<TypeParam>(basis, cache);
            ASSERT_NO_FATAL_FAILURE(expect_canonical_csc(l2t, tensor_size, lie_size));
            ASSERT_NO_FATAL_FAILURE(expect_canonical_csc(t2l, lie_size, tensor_size));

            for (Index col = 0; col < lie_size; ++col) {
                std::vector<double> result(lie_size, 0);
                for (auto i = l2t.offsets()[col]; i < l2t.offsets()[col + 1]; ++i) {
                    const auto tensor_row = l2t.indices()[i];
                    for (auto j = t2l.offsets()[tensor_row];
                         j < t2l.offsets()[tensor_row + 1]; ++j) {
                        result[t2l.indices()[j]] +=
                            static_cast<double>(l2t.data()[i]) * t2l.data()[j];
                    }
                }
                for (Index row = 0; row < lie_size; ++row) {
                    EXPECT_NEAR(result[row], row == col ? 1 : 0, tolerance)
                        << "row " << row << ", column " << col;
                }
            }
        }
    }
}

} // namespace
