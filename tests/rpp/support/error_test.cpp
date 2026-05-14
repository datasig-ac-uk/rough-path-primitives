#include <gtest/gtest.h>
#include "rpp/support/error.hpp"

using namespace rpp;

TEST(ErrorTest, SuccessMessage) {
    Error<void> e(ErrorCode::Success);
    EXPECT_TRUE(e);
    EXPECT_FALSE(!e);
    EXPECT_EQ(e.message(), std::string_view("Success"));
}

TEST(ErrorTest, CancelledMessage) {
    Error<void> e(ErrorCode::Cancelled);
    EXPECT_FALSE(static_cast<bool>(e));
    EXPECT_FALSE(e);
    EXPECT_EQ(e.message(), std::string_view("Operation cancelled"));
}

TEST(ErrorTest, UnknownMessage) {
    Error<void> e(ErrorCode::Unknown);
    EXPECT_FALSE(static_cast<bool>(e));
    EXPECT_FALSE(e);
    EXPECT_EQ(e.message(), std::string_view("Unknown error"));
}

TEST(ErrorTest, PayloadStringMessage) {
    Error<std::string> e(ErrorCode::InvalidArgument, "bad arg");
    EXPECT_FALSE(static_cast<bool>(e));
    EXPECT_FALSE(e);
    EXPECT_EQ(e.message(), "bad arg");
}

TEST(ErrorTest, PayloadCStringMessage) {
    Error<const char*> e(ErrorCode::Timeout, "timeout");
    EXPECT_FALSE(static_cast<bool>(e));
    EXPECT_FALSE(e);
    EXPECT_EQ(e.message(), "timeout");
}