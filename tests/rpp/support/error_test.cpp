#include <gtest/gtest.h>
#include "rpp/support/error.hpp"

using namespace rpp;

TEST(ErrorTest, OkMessage) {
    Error<void> e(ErrorCode::Ok);
    EXPECT_TRUE(!e);
    EXPECT_EQ(e.message(), std::string_view("Success"));
}

TEST(ErrorTest, CancelledMessage) {
    Error<void> e(ErrorCode::Cancelled);
    EXPECT_TRUE(e);
    EXPECT_EQ(e.message(), std::string_view("Operation cancelled"));
}

TEST(ErrorTest, UnknownMessage) {
    Error<void> e(ErrorCode::Unknown);
    EXPECT_TRUE(e);
    EXPECT_EQ(e.message(), std::string_view("Unknown error"));
}

TEST(ErrorTest, PayloadStringMessage) {
    Error<std::string> e(ErrorCode::InvalidArgument, "bad arg");
    EXPECT_TRUE(e);
    EXPECT_EQ(e.message(), "bad arg");
}

TEST(ErrorTest, PayloadCStringMessage) {
    Error<const char*> e(ErrorCode::Timeout, "timeout");
    EXPECT_TRUE(e);
    EXPECT_EQ(e.message(), "timeout");
}