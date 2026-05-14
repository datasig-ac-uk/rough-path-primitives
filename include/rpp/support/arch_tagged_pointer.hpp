#ifndef RPP_SUPPORT_ARCH_TAGGED_POINTER_HPP
#define RPP_SUPPORT_ARCH_TAGGED_POINTER_HPP

#include <rpp/config.h>

#include <rpp/support/iterator_traits.hpp>

namespace rpp {
template<typename T, typename Arch>
class Ptr {
    T *ptr_;

public:
    using Architecture = Arch;
    using value_type = T;
    using pointer = T *;
    using reference = T &;
    using const_pointer = std::add_const_t<T> *;
    using const_reference = std::add_const_t<T> &;
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = typename Architecture::Index;
    using size_type = typename Architecture::Size;

    constexpr Ptr(T *ptr) : ptr_(ptr) {
    }

    constexpr Ptr(nullptr_t nptr RPP_MAYBE_UNUSED) : ptr_(nullptr) {
    }

    template <typename U, typename=std::enable_if_t<std::is_convertible_v<T*, U*>>>
    explicit constexpr operator U* () const noexcept {
        return static_cast<U*>(ptr_);
    }

    constexpr pointer raw_ptr() const noexcept { return ptr_; }

    constexpr reference operator*() const { return *ptr_; }
    constexpr pointer operator->() const { return ptr_; }

    constexpr reference operator[](difference_type i) const { return ptr_[i]; }

    constexpr Ptr &operator++() {
        ++ptr_;
        return *this;
    }

    constexpr Ptr operator++(int) {
        Ptr tmp(*this);
        ++(*this);
        return tmp;
    }

    constexpr Ptr &operator--() {
        --ptr_;
        return *this;
    }

    constexpr Ptr operator--(int) {
        Ptr tmp(*this);
        --(*this);
        return tmp;
    }

    constexpr Ptr &operator+=(difference_type i) {
        ptr_ += i;
        return *this;
    }

    constexpr Ptr &operator-=(difference_type i) {
        ptr_ -= i;
        return *this;
    }

    friend constexpr bool operator==(Ptr const &lhs, Ptr const &rhs) { return lhs.ptr_ == rhs.ptr_; }
    friend constexpr bool operator!=(Ptr const &lhs, Ptr const &rhs) { return lhs.ptr_ != rhs.ptr_; }
    friend constexpr bool operator<(Ptr const &lhs, Ptr const &rhs) { return lhs.ptr_ < rhs.ptr_; }
    friend constexpr bool operator>(Ptr const &lhs, Ptr const &rhs) { return lhs.ptr_ > rhs.ptr_; }
    friend constexpr bool operator<=(Ptr const &lhs, Ptr const &rhs) { return lhs.ptr_ <= rhs.ptr_; }
    friend constexpr bool operator>=(Ptr const &lhs, Ptr const &rhs) { return lhs.ptr_ >= rhs.ptr_; }

    friend constexpr Ptr operator+(Ptr const &lhs, difference_type rhs) { return Ptr(lhs.ptr_ + rhs); }
    friend constexpr Ptr operator-(Ptr const &lhs, difference_type rhs) { return Ptr(lhs.ptr_ - rhs); }
    friend constexpr difference_type operator-(Ptr const &lhs, Ptr const &rhs) { return lhs.ptr_ - rhs.ptr_; }
};


template <typename Arch, typename T>
Ptr<T, Arch> tag_pointer(T* ptr) {
    return Ptr<T, Arch>(ptr);
}

namespace traits {

template <typename T, typename Arch>
struct IteratorTraits<Ptr<T, Arch>, void> : public std::iterator_traits<Ptr<T *, Arch>> {
    using Architecture = Arch;
};


template <typename Ptr>
inline constexpr bool is_arch_ptr_v = false;

template <typename T, typename Arch>
inline constexpr bool is_arch_ptr_v<Ptr<T, Arch>> = true;




}// namespace traits
} // namespace rpp

#endif //RPP_SUPPORT_ARCH_TAGGED_POINTER_HPP
