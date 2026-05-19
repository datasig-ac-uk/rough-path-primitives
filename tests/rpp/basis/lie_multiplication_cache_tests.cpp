#include <gtest/gtest.h>

#include <rpp/basis/hall_basis.hpp>
#include <rpp/basis/lie_multiplication.hpp>

namespace {

using Architecture = rpp::arch::NativeArchitecture;
using Degree = Architecture::Degree;
using Cache = rpp::basis::LieMultiplicationCache<Architecture>;
using CacheEntry = Cache::CacheEntry;

[[nodiscard]] auto make_lie_basis(Degree width = 3, Degree depth = 4)
{
    return rpp::basis::HallBasis<Architecture>(width, depth);
}

TEST(LieMultiplicationCacheTests, ReturnsDirectAndAntisymmetricBasisProducts)
{
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_EQ(cache.get_bracket(1, 2), (CacheEntry{{4, 1}}));
    EXPECT_EQ(cache.get_bracket(2, 1), (CacheEntry{{4, -1}}));
    EXPECT_TRUE(cache.get_bracket(1, 1).empty());
}

TEST(LieMultiplicationCacheTests, ReturnsEmptyProductOutsideBasisDepth)
{
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_TRUE(cache.get_bracket(1, 33).empty());
    EXPECT_TRUE(cache.get_bracket(1, 15).empty());
    EXPECT_TRUE(cache.get_bracket(0, 1).empty());
}

TEST(LieMultiplicationCacheTests, ExpandsNonHallBracketByJacobiIdentity)
{
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_EQ(cache.get_bracket(1, 6), (CacheEntry{{10, 1}, {12, -1}}));
    EXPECT_EQ(cache.get_bracket(6, 1), (CacheEntry{{10, -1}, {12, 1}}));
}

TEST(LieMultiplicationCacheTests, ExpandsRecursiveNonHallBracket)
{
    auto const hall = make_lie_basis();
    Cache cache(hall.to_lie_basis());

    EXPECT_EQ(cache.get_bracket(1, 9), (CacheEntry{{17, 1}}));
    EXPECT_EQ(cache.get_bracket(9, 1), (CacheEntry{{17, -1}}));
}

} // namespace
