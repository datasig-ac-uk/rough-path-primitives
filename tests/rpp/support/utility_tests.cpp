#include <cstdint>

#include <gtest/gtest.h>

#include <rpp/utility.hpp>

namespace {

TEST(UtilityTests, IntegerPowersHandleOddAndEvenExponents) {
    static_assert(rpp::const_power(3, 0) == 1);
    static_assert(rpp::const_power(3, 1) == 3);
    static_assert(rpp::const_power(3, 2) == 9);
    static_assert(rpp::const_power(3, 3) == 27);
    static_assert(rpp::const_power(3, 4) == 81);
    EXPECT_EQ(rpp::const_power(2, 5), 32);
    EXPECT_EQ(rpp::const_power(-2, 3), -8);
    EXPECT_EQ(rpp::const_power(0, 3), 0);
    // Do not square the base once the requested exponent is exhausted.
    EXPECT_EQ(rpp::const_power(std::int64_t{2}, 62), std::int64_t{1} << 62);
}

} // namespace
