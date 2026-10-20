#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <rpp/basis/hall_basis.hpp>
#include <rpp/basis/lie_multiplication.hpp>

namespace {

using Architecture = rpp::arch::NativeArchitecture;
using Degree = Architecture::Degree;
using Index = Architecture::Index;
using Cache = rpp::basis::LieMultiplicationCache<Architecture>;
using CacheEntry = std::vector<Cache::CacheEntry::value_type>;

CacheEntry copy_entry(Cache::CacheEntry entry) {
    if (entry.empty()) {
        return {};
    }
    return {entry.begin(), entry.end()};
}

[[nodiscard]] auto make_lie_basis(Degree width = 3, Degree depth = 4) {
    return rpp::basis::HallBasis<Architecture>(width, depth);
}

TEST(LieMultiplicationCacheTests, ReturnsDirectAndAntisymmetricBasisProducts) {
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_EQ(copy_entry(cache.get_bracket(1, 2)), (CacheEntry{{4, 1}}));
    EXPECT_EQ(copy_entry(cache.get_bracket(2, 1)), (CacheEntry{{4, -1}}));
    EXPECT_TRUE(cache.get_bracket(1, 1).empty());
}

TEST(LieMultiplicationCacheTests, ReturnsEmptyProductOutsideBasisDepth) {
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_TRUE(cache.get_bracket(1, 33).empty());
    EXPECT_TRUE(cache.get_bracket(1, 15).empty());
    EXPECT_TRUE(cache.get_bracket(0, 1).empty());
}

TEST(LieMultiplicationCacheTests, ExpandsNonHallBracketByJacobiIdentity) {
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_EQ(copy_entry(cache.get_bracket(1, 6)), (CacheEntry{{10, 1}, {12, -1}}));
    EXPECT_EQ(copy_entry(cache.get_bracket(6, 1)), (CacheEntry{{10, -1}, {12, 1}}));
}

TEST(LieMultiplicationCacheTests, ExpandsRecursiveNonHallBracket) {
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_EQ(copy_entry(cache.get_bracket(1, 9)), (CacheEntry{{17, 1}}));
    EXPECT_EQ(copy_entry(cache.get_bracket(9, 1)), (CacheEntry{{17, -1}}));
}

TEST(LieMultiplicationCacheTests, EliminatesSelfBracketsInJacobiExpansion) {
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    // [x,[y,[x,y]]] = [[x,y],[x,y]] + [y,[x,[x,y]]].
    // The first term vanishes, leaving a single Hall word.
    EXPECT_EQ(copy_entry(cache.get_bracket(1, 9)), (CacheEntry{{17, 1}}));
    // The same cancellation occurs with z in place of y.
    EXPECT_EQ(copy_entry(cache.get_bracket(1, 13)), (CacheEntry{{23, 1}}));
}

TEST(LieMultiplicationCacheTests, CombinesRecursiveContributionsWithMultiplicity) {
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    // [x,[y,[y,z]]] includes two copies of [[x,y],[y,z]].
    EXPECT_EQ(copy_entry(cache.get_bracket(1, 11)),
              (CacheEntry{{20, 1}, {24, -1}, {31, 2}}));
    EXPECT_EQ(copy_entry(cache.get_bracket(11, 1)),
              (CacheEntry{{20, -1}, {24, 1}, {31, -2}}));
    EXPECT_EQ(copy_entry(cache.get_bracket(1, 14)),
              (CacheEntry{{25, 1}, {27, -1}, {32, 1}}));
}

// An independent oracle in the free associative algebra: expand each Hall
// tree using concatenation and subtraction, without Jacobi or bracket lookup.
using TensorExpansion = std::map<std::string, std::int64_t>;

void remove_zeros(TensorExpansion& expansion) {
    for (auto it = expansion.begin(); it != expansion.end();) {
        if (it->second == 0) {
            it = expansion.erase(it);
        }
        else {
            ++it;
        }
    }
}

TensorExpansion tensor_commutator(TensorExpansion const& left,
                                  TensorExpansion const& right) {
    TensorExpansion result;
    for (auto const& [u, a] : left) {
        for (auto const& [v, b] : right) {
            result[u + v] += a * b;
            result[v + u] -= b * a;
        }
    }
    remove_zeros(result);
    return result;
}

void expect_all_brackets(Degree width, Degree depth) {
    auto const hall = make_lie_basis(width, depth);
    auto const basis = hall.to_lie_basis();
    std::vector<TensorExpansion> tensors(basis.true_size());
    for (Index key = 1; key < basis.true_size(); ++key) {
        auto const [left, right] = basis[key];
        if (left == 0) {
            tensors[key][std::string(1, static_cast<char>('a' + right - 1))] = 1;
        }
        else {
            tensors[key] = tensor_commutator(tensors[left], tensors[right]);
        }
    }

    Cache warm_cache(basis);
    // Include the zero key, equal factors, both orientations, all degree
    // splits, and products beyond the truncation depth.
    for (Index left = 0; left < basis.true_size(); ++left) {
        for (Index right = 0; right < basis.true_size(); ++right) {
            SCOPED_TRACE(::testing::Message() << "bracket (" << left << ", "
                                             << right << ")");
            const auto degree = basis.degree(left) + basis.degree(right);
            auto const expected = degree <= basis.depth
                ? tensor_commutator(tensors[left], tensors[right])
                : TensorExpansion{};

            Cache cold_cache(basis);
            auto const cold_result = copy_entry(cold_cache.get_bracket(left, right));
            auto const result = warm_cache.get_bracket(left, right);
            EXPECT_EQ(copy_entry(result), cold_result);

            TensorExpansion actual;
            Index previous = 0;
            for (auto const& [key, coefficient] : result) {
                ASSERT_GT(key, previous);
                ASSERT_LT(key, basis.true_size());
                EXPECT_NE(coefficient, 0);
                EXPECT_EQ(basis.degree(key), degree);
                previous = key;
                for (auto const& [word, value] : tensors[key]) {
                    actual[word] += coefficient * value;
                }
            }
            remove_zeros(actual);
            EXPECT_EQ(actual, expected);
            // A repeated request must preserve the completed result.
            EXPECT_EQ(copy_entry(warm_cache.get_bracket(left, right)), cold_result);
        }
    }
}

TEST(LieMultiplicationCacheTests, AllBracketsAgreeWithTensorCommutators) {
    expect_all_brackets(3, 4);
}

TEST(LieMultiplicationCacheTests, DeeperRecursionAgreesWithTensorCommutators) {
    // Exercise longer Jacobi chains and multi-term outer expansions while
    // fresh caches repeatedly grow their shared coefficient storage.
    expect_all_brackets(3, 6);
}

} // namespace
