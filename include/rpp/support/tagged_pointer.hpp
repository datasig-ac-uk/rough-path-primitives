#ifndef RPP_SUPPORT_TAGGED_POINTER_HPP
#define RPP_SUPPORT_TAGGED_POINTER_HPP

#include <type_traits>
#include <utility>

#include <rpp/config.h>

#include <rpp/support/iterator_traits.hpp>

namespace rpp {

namespace tags {

struct HostLocation {};

template <typename Loc>
struct LocationTag {
    using Location = Loc;
};


namespace detail {

template <typename T, typename = void>
struct LocationOfImpl {
    using type = HostLocation;
};

template <typename T>
struct LocationOfImpl<T, std::void_t<typename T::Location>> {
    using type = typename T::Location;
};

} // namespace detail

template <typename T>
using location_of_t = typename detail::LocationOfImpl<T>::type;

} // namespace tags

template <typename T, typename... Tags>
class TaggedPtr : public Tags... {
    static_assert((std::is_empty_v<Tags> && ... && true),
                  "Tag classes must be empty");
    T* ptr_;

public:
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using const_pointer = std::add_const_t<T>*;
    using const_reference = std::add_const_t<T>&;
    using iterator_category = std::random_access_iterator_tag;


    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    RPP_HOST_DEVICE constexpr TaggedPtr(U* ptr)
        : Tags{}..., ptr_(static_cast<T*>(ptr)) {}


    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<T*, U*>>>
    RPP_HOST_DEVICE explicit constexpr operator U*() const noexcept {
        return static_cast<U*>(ptr_);
    }

    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<T*, U*>>>
    RPP_HOST_DEVICE explicit constexpr
    operator TaggedPtr<U, Tags...>() const noexcept {
        return TaggedPtr<U, Tags...>(ptr_);
    }

    RPP_HOST_DEVICE constexpr pointer raw_ptr() const noexcept { return ptr_; }

    RPP_HOST_DEVICE constexpr reference operator*() const { return *ptr_; }
    RPP_HOST_DEVICE constexpr pointer operator->() const { return ptr_; }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    RPP_HOST_DEVICE constexpr reference operator[](I i) const {
        return ptr_[i];
    }

    RPP_HOST_DEVICE constexpr TaggedPtr& operator++() {
        ++ptr_;
        return *this;
    }

    RPP_HOST_DEVICE constexpr TaggedPtr operator++(int) {
        TaggedPtr tmp(*this);
        ++(*this);
        return tmp;
    }

    RPP_HOST_DEVICE constexpr TaggedPtr& operator--() {
        --ptr_;
        return *this;
    }

    RPP_HOST_DEVICE constexpr TaggedPtr operator--(int) {
        TaggedPtr tmp(*this);
        --(*this);
        return tmp;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    RPP_HOST_DEVICE constexpr TaggedPtr& operator+=(I i) {
        ptr_ += i;
        return *this;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    RPP_HOST_DEVICE constexpr TaggedPtr& operator-=(I i) {
        ptr_ -= i;
        return *this;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator==(TaggedPtr const& lhs,
                                     TaggedPtr const& rhs) {
        return lhs.ptr_ == rhs.ptr_;
    }
    RPP_HOST_DEVICE
    friend constexpr bool operator!=(TaggedPtr const& lhs,
                                     TaggedPtr const& rhs) {
        return lhs.ptr_ != rhs.ptr_;
    }
    RPP_HOST_DEVICE
    friend constexpr bool operator<(TaggedPtr const& lhs,
                                    TaggedPtr const& rhs) {
        return lhs.ptr_ < rhs.ptr_;
    }
    RPP_HOST_DEVICE
    friend constexpr bool operator>(TaggedPtr const& lhs,
                                    TaggedPtr const& rhs) {
        return lhs.ptr_ > rhs.ptr_;
    }
    RPP_HOST_DEVICE
    friend constexpr bool operator<=(TaggedPtr const& lhs,
                                     TaggedPtr const& rhs) {
        return lhs.ptr_ <= rhs.ptr_;
    }
    RPP_HOST_DEVICE
    friend constexpr bool operator>=(TaggedPtr const& lhs,
                                     TaggedPtr const& rhs) {
        return lhs.ptr_ >= rhs.ptr_;
    }

    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    RPP_HOST_DEVICE friend constexpr TaggedPtr operator+(TaggedPtr const& lhs,
                                                         I rhs) {
        return TaggedPtr(lhs.ptr_ + rhs);
    }
    template <typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    RPP_HOST_DEVICE friend constexpr TaggedPtr operator-(TaggedPtr const& lhs,
                                                         I rhs) {
        return TaggedPtr(lhs.ptr_ - rhs);
    }
    RPP_HOST_DEVICE
    friend constexpr auto operator-(TaggedPtr const& lhs,
                                    TaggedPtr const& rhs) {
        return lhs.ptr_ - rhs.ptr_;
    }
};


template <typename T, typename... Tags>
TaggedPtr<T, Tags...> tag_pointer(T* ptr, Tags...) {
    return TaggedPtr<T, Tags...>(ptr);
}


template <typename T, typename... Tags>
RPP_HOST_DEVICE constexpr T*
raw_pointer_cast(TaggedPtr<T, Tags...> ptr) noexcept {
    return ptr.raw_ptr();
}


namespace traits {

template <typename T, typename P>
inline constexpr bool is_tagged_ptr_for_v = false;

template <typename T, typename... Tags>
inline constexpr bool is_tagged_ptr_for_v<T, TaggedPtr<T, Tags...>> = true;

} // namespace traits
} // namespace rpp

#endif // RPP_SUPPORT_TAGGED_POINTER_HPP
