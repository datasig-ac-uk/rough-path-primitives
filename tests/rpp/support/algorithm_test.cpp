#include <gtest/gtest.h>
#include "rpp/support/algorithm.hpp"

using namespace rpp::algo;

TEST(AlgorithmTest, LowerBoundBasic) {
    int data[] = {1, 3, 5, 7};
    const std::size_t n = std::size(data);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 0), 0);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 1), 0);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 2), 1);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 3), 1);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 4), 2);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 7), 3);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 8), 4);
}

TEST(AlgorithmTest, UpperBoundBasic) {
    int data[] = {1, 3, 5, 7};
    const std::size_t n = std::size(data);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 0), 0);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 1), 1);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 2), 1);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 3), 2);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 4), 2);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 7), 4);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 8), 4);
}


TEST(AlgorithmTest, CustomComparator) {
    // descending order with std::greater
    int data[] = {7, 5, 3, 1};
    const std::size_t n = std::size(data);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 6, std::greater<>{}), 1); // first >=6 in descending: 7 at 0
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 5, std::greater<>{}), 1);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 4, std::greater<>{}), 2);
    EXPECT_EQ(index_lower_bound(data, size_t{0}, n, 0, std::greater<>{}), 4);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 6, std::greater<>{}), 1); // first >6 is 5 at index1
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 5, std::greater<>{}), 2);
    EXPECT_EQ(index_upper_bound(data, size_t{0}, n, 0, std::greater<>{}), 4);
}
