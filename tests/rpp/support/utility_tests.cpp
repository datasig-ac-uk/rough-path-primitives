#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include <rpp/utility.hpp>

namespace {

TEST(UtilityTests, AssumeAlignedPreservesAddressAndPointerQualifiers) {
    alignas(64) double data[8]{};
    const double* const_ptr = data;
    volatile double* volatile_ptr = data;
    const volatile double* cv_ptr = data;
    void* void_ptr = data;
    static_assert(std::is_same_v<decltype(rpp::assume_aligned<64>(const_ptr)),
                                 const double*>);
    static_assert(std::is_same_v<decltype(rpp::assume_aligned<64>(volatile_ptr)),
                                 volatile double*>);
    static_assert(std::is_same_v<decltype(rpp::assume_aligned<64>(cv_ptr)),
                                 const volatile double*>);
    EXPECT_EQ(rpp::assume_aligned<64>(data), data);
    EXPECT_EQ(rpp::assume_aligned<64>(const_ptr), const_ptr);
    EXPECT_EQ(rpp::assume_aligned<64>(volatile_ptr), volatile_ptr);
    EXPECT_EQ(rpp::assume_aligned<64>(cv_ptr), cv_ptr);
    EXPECT_EQ(rpp::assume_aligned<64>(void_ptr), void_ptr);
}

#ifndef NDEBUG
TEST(UtilityTests, AssumeAlignedRejectsMisalignedPointersInDebugBuilds) {
    alignas(64) unsigned char data[64]{};
    EXPECT_DEATH_IF_SUPPORTED(
        (void)rpp::assume_aligned<64>(data + 1), "assumed alignment");
}
#endif

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
