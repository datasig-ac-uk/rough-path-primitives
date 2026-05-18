#ifndef INCLUDE_RPP_SUPPORT_STRIDED_ITERATOR_HPP
#define INCLUDE_RPP_SUPPORT_STRIDED_ITERATOR_HPP

#include <type_traits>
#include <iterator>

#include <rpp/config.h>

#include <rpp/support/iterator_traits.hpp>
#include <rpp/architecture.hpp>

namespace rpp {


template <typename It>
class StridedIterator {
    static_assert(traits::is_random_access_v<It>, "base iterator must support random access");

    using Traits = traits::IteratorTraits<It>;
    It base_;
    typename Traits::difference_type stride_;

public:
    using Architecture = traits::arch_of_t<It>;
    using Location = traits::location_of_t<It>;
    using value_type = typename Traits::value_type;
    using difference_type = typename Traits::difference_type;
    using pointer = typename Traits::pointer;
    using reference = typename Traits::reference;
    using iterator_category = std::random_access_iterator_tag;

    RPP_HOST_DEVICE
    constexpr StridedIterator(It base, difference_type stride) noexcept
        : base_(base), stride_(stride)
    {}

    RPP_HOST_DEVICE
    constexpr StridedIterator(It base, difference_type stride, difference_type offset) noexcept
        : base_(base + offset * stride), stride_(stride)
    {}

    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator*() const noexcept {
        return *base_;
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator->() const noexcept {
        return base_;
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference operator[](difference_type i) const noexcept {
        return base_[i * stride_];
    }

    RPP_HOST_DEVICE
    constexpr StridedIterator& operator++() noexcept {
        base_ += stride_;
        return *this;
    }

    RPP_HOST_DEVICE
    constexpr StridedIterator operator++(int) noexcept {
        StridedIterator tmp(*this);
        ++(*this);
        return tmp;
    }

    RPP_HOST_DEVICE
    constexpr StridedIterator& operator--() noexcept {
        base_ -= stride_;
        return *this;
    }

    RPP_HOST_DEVICE
    constexpr StridedIterator operator--(int) noexcept {
        StridedIterator tmp(*this);
        --(*this);
        return tmp;
    }

    RPP_HOST_DEVICE
    constexpr StridedIterator& operator+=(difference_type offset) noexcept {
        base_ += offset * stride_;
        return *this;
    }

    RPP_HOST_DEVICE
    constexpr StridedIterator& operator-=(difference_type offset) noexcept {
        base_ -= offset * stride_;
        return *this;
    }

    RPP_HOST_DEVICE
    friend constexpr StridedIterator operator+(StridedIterator const& iter, difference_type offset) noexcept {
        return { iter.base_, iter.stride_, offset };
    }

    RPP_HOST_DEVICE
    friend constexpr StridedIterator operator+(difference_type offset, StridedIterator const& iter) noexcept {
        return { iter.base_, iter.stride_, offset };
    }

    RPP_HOST_DEVICE
    friend constexpr StridedIterator operator-(StridedIterator const& iter, difference_type offset) noexcept {
        return { iter.base_, iter.stride_, -offset };
    }

    RPP_HOST_DEVICE
    friend constexpr difference_type operator-(StridedIterator const& iter, StridedIterator const& other) noexcept {
        return static_cast<difference_type>(iter.base_ - other.base_) / static_cast<difference_type>(other.stride_);
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator==(StridedIterator const& iter, StridedIterator const& other) noexcept {
        return iter.base_ == other.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator==(StridedIterator const& iter, It const& other) noexcept {
        return iter.base_ == other;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator==(It const& other, StridedIterator const& iter) noexcept {
        return other == iter.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator!=(StridedIterator const& iter, StridedIterator const& other) noexcept {
        return iter.base_ != other.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator!=(StridedIterator const& iter, It const& other) noexcept {
        return iter.base_ != other;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator!=(It const& other, StridedIterator const& iter) noexcept {
        return other != iter.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<(StridedIterator const& iter, StridedIterator const& other) noexcept {
        return iter.base_ < other.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<(StridedIterator const& iter, It const& other) noexcept {
        return iter.base_ < other;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<(It const& other, StridedIterator const& iter) noexcept {
        return other < iter.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<=(StridedIterator const& iter, StridedIterator const& other) noexcept {
        return iter.base_ <= other.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<=(StridedIterator const& iter, It const& other) noexcept {
        return iter.base_ <= other;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<=(It const& other, StridedIterator const& iter) noexcept {
        return other <= iter.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator>(StridedIterator const& iter, StridedIterator const& other) noexcept {
        return iter.base_ > other.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<(StridedIterator const& iter, It const& other) noexcept {
        return iter.base_ > other;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator<(It const& other, StridedIterator const& iter) noexcept {
        return other > iter.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator>=(StridedIterator const& iter, StridedIterator const& other) noexcept {
        return iter.base_ >= other.base_;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator>=(StridedIterator const& iter, It const& other) noexcept {
        return iter.base_ >= other;
    }

    RPP_HOST_DEVICE
    friend constexpr bool operator>=(It const& other, StridedIterator const& iter) noexcept {
        return other >= iter.base_;
    }

    RPP_HOST_DEVICE
    constexpr It base() const noexcept { return base_; }

    RPP_HOST_DEVICE
    constexpr It base_at(difference_type offset) const noexcept {
        return base_ + stride_*offset;
    }


};

namespace traits {

template <typename It>
struct IteratorTraits<StridedIterator<It>> : IteratorTraits<It> {
};

}


} // namespace rpp

#endif //INCLUDE_RPP_SUPPORT_STRIDED_ITERATOR_HPP
