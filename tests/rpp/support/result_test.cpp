#include <gtest/gtest.h>
#include <rpp/support/error.hpp>

using namespace rpp;


TEST(ResultTest, PayloadOk) {
    Result<std::string, Error<void>> okStr(std::string("hello"));
    EXPECT_TRUE(okStr.ok());
    EXPECT_EQ(okStr.value(), std::string("hello"));
}

TEST(ResultTest, PayloadError) {
    Result<std::string, Error<void>> errStr{Error<void>(ErrorCode::Timeout)};
    EXPECT_FALSE(errStr.ok());
    EXPECT_EQ(errStr.error().code(), ErrorCode::Timeout);
}
