#include <gtest/gtest.h>

#include <rpp/basis/hall_basis.hpp>

namespace {

using Architecture = rpp::arch::NativeArchitecture;
using Degree = Architecture::Degree;
using Index = Architecture::Index;

[[nodiscard]] auto make_lie_basis(Degree width = 3, Degree depth = 4) {
    return rpp::basis::HallBasis<Architecture>(width, depth);
}

TEST(HallLieBasisTests, GeneratesExpectedDegreeRanges) {
    auto const hall = make_lie_basis();
    auto const basis = hall.to_lie_basis();

    EXPECT_EQ(basis.true_size(), Index{33});
    EXPECT_EQ(basis.size(), Index{32});

    EXPECT_EQ(basis.start_of_degree(0), Index{0});
    EXPECT_EQ(basis.end_of_degree(0), Index{1});
    EXPECT_EQ(basis.start_of_degree(1), Index{1});
    EXPECT_EQ(basis.end_of_degree(1), Index{4});
    EXPECT_EQ(basis.start_of_degree(2), Index{4});
    EXPECT_EQ(basis.end_of_degree(2), Index{7});
    EXPECT_EQ(basis.start_of_degree(3), Index{7});
    EXPECT_EQ(basis.end_of_degree(3), Index{15});
    EXPECT_EQ(basis.start_of_degree(4), Index{15});
    EXPECT_EQ(basis.end_of_degree(4), Index{33});
}

TEST(HallLieBasisTests, FindsDirectHallBracketsByPair) {
    auto const hall = make_lie_basis();
    auto const basis = hall.to_lie_basis();

    EXPECT_EQ(basis.find_bracket(1, 2, 2), Index{4});
    EXPECT_EQ(basis.find_bracket(1, 2), Index{4});
    EXPECT_EQ(basis.find_bracket(2, 5, 3), Index{10});
    EXPECT_EQ(basis.find_bracket(3, 4, 3), Index{12});
}

TEST(HallLieBasisTests, ReturnsGodIndexForPairsWithNoDirectBasisElement) {
    auto const hall = make_lie_basis();
    auto const basis = hall.to_lie_basis();

    EXPECT_EQ(basis.find_bracket(2, 1, 2), Index{0});
    EXPECT_EQ(basis.find_bracket(1, 1, 2), Index{0});

    // [1, [2, 3]] has a non-trivial Hall expansion but is not itself a basis
    // element.
    EXPECT_EQ(basis.find_bracket(1, 6, 3), Index{0});
    EXPECT_EQ(basis.find_bracket(1, 9, 4), Index{0});
}

} // namespace
