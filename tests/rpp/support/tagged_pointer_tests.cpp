#include <gtest/gtest.h>
#include <rpp/architecture.hpp>
#include <rpp/support/tagged_pointer.hpp>
#include <type_traits>

using namespace rpp;

// Test dereferencing and element access
TEST(TaggedPointerTest, Dereference) {
    int data[5] = {10, 20, 30, 40, 50};
    using Ptr = arch::NativeArchitecture::Ptr<int>;
    Ptr p(&data[0]);
    EXPECT_EQ(*p, 10);
    EXPECT_EQ(p[2], 30);
}

// Test increment and decrement operators
TEST(TaggedPointerTest, IncrementDecrement) {
    int data[5] = {10, 20, 30, 40, 50};
    using Ptr = arch::NativeArchitecture::Ptr<int>;
    Ptr p(&data[0]);
    ++p;
    EXPECT_EQ(*p, 20);
    p++;
    EXPECT_EQ(*p, 30);
    --p;
    EXPECT_EQ(*p, 20);
    p--;
    EXPECT_EQ(*p, 10);
}

// Test pointer arithmetic operations
TEST(TaggedPointerTest, Arithmetic) {
    int data[5] = {10, 20, 30, 40, 50};
    using Ptr = arch::NativeArchitecture::Ptr<int>;
    Ptr p(&data[0]);
    Ptr q = p + 3; // points to data[3]
    EXPECT_EQ(*q, 40);
    q = q - 1; // now points to data[2]
    EXPECT_EQ(*q, 30);
    EXPECT_EQ(q - p, 2);
}

// Test comparison operators
TEST(TaggedPointerTest, Comparisons) {
    int data[5] = {10, 20, 30, 40, 50};
    using Ptr = arch::NativeArchitecture::Ptr<int>;
    Ptr p(&data[0]);
    Ptr q = p + 3;
    EXPECT_TRUE(p < q);
    EXPECT_TRUE(q > p);
    EXPECT_TRUE(p <= p);
    EXPECT_TRUE(q >= p);
    EXPECT_TRUE(p != q);
    EXPECT_TRUE(p == p);
    // raw_pointer_cast sanity check
    EXPECT_EQ(raw_pointer_cast(p), &data[0]);
}

namespace {

struct MyArchitecture {
    using Index = int;
    using Size = unsigned;
    using Degree = int;

    template <typename T>
    using Ptr = TaggedPtr<T,
                          tags::ArchTag<MyArchitecture>,
                          tags::LocationTag<HostLocation>>;
};

} // namespace

// Test tag inheritance and architecture traits
TEST(TaggedPointerTest, TagInheritanceAndTraits) {
    using Ptr = MyArchitecture::Ptr<int>;

    static_assert(std::is_same_v<std::iterator_traits<Ptr>::difference_type,
                                 typename MyArchitecture::Index>,
                  "difference type should be set by architecture");

    static_assert(std::is_base_of_v<tags::ArchTag<MyArchitecture>, Ptr>,
                  "Ptr should inherit ArchTag");
    static_assert(std::is_base_of_v<tags::LocationTag<HostLocation>, Ptr>,
                  "Ptr should inherit LocationTag");
    static_assert(std::is_same_v<traits::arch_of_t<Ptr>, MyArchitecture>,
                  "arch_of_t should give the architecture type");
    SUCCEED();
}
