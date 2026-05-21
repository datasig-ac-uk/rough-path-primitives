#ifndef RPP_SUPPORT_SPAN_HPP
#define RPP_SUPPORT_SPAN_HPP

#include <array>
#include <cstddef>
#include <iterator>
#include <limits>
#include <span>
#include <type_traits>

#ifdef __cpp_lib_span
#include <span>
#endif

#include <rpp/config.h>

namespace rpp {

#ifdef __cpp_lib_span
template <typename T, size_t N = std::dynamic_extent>
using Span = std::span<T, N>;
using std::dynamic_extent;
#else
inline constexpr size_t dynamic_extent = std::numeric_limits<size_t>::max();

namespace detail {

template <size_t N>
struct ExtentHolder {
    static constexpr auto size_ = N;

    RPP_HOST_DEVICE
    explicit constexpr ExtentHolder(size_t size) noexcept {}

    RPP_HOST_DEVICE RPP_NODISCARD constexpr size_t size() const noexcept {
        return size_;
    }
};

template <>
struct ExtentHolder<dynamic_extent> {
    size_t size_ = 0;

    RPP_HOST_DEVICE
    explicit constexpr ExtentHolder(size_t size) noexcept : size_(size) {}

    RPP_HOST_DEVICE RPP_NODISCARD constexpr size_t size() const noexcept {
        return size_;
    }
};


} // namespace detail

template <typename T, size_t N = dynamic_extent>
class Span : detail::ExtentHolder<N> {
    using Extent = detail::ExtentHolder<N>;
    T* data_ = nullptr;

    template <size_t OtherExtent, size_t Offset = 0>
    static constexpr size_t subspan_extent() noexcept {
        static_assert(
            Offset <= N &&
                (OtherExtent == dynamic_extent || OtherExtent <= N - Offset),
            "subspan is malformed");
        if constexpr (OtherExtent != dynamic_extent) {
            return OtherExtent;
        }
        else if constexpr (N != dynamic_extent) {
            return N - Offset;
        }
        else {
            return dynamic_extent;
        }
    }

public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using reference = T&;
    using const_reference = std::add_const_t<T>&;
    using iterator = T*;
    using const_iterator = std::add_const_t<T>*;
    using pointer = T*;
    using const_pointer = std::add_const_t<T>*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;

    static constexpr size_t extent = N;

    RPP_HOST_DEVICE
    constexpr Span(T* data, size_t size) : Extent(size), data_(data) {}

    RPP_HOST_DEVICE
    constexpr Span(T* data) : Extent(N), data_(data) {
        static_assert(N != dynamic_extent, "unspecified size");
    }

    template <typename Container>
    RPP_HOST_DEVICE constexpr Span(Container& container) noexcept
        : Extent(container.size()), data_(container.data()) {}

    template <typename Container>
    RPP_HOST_DEVICE constexpr Span(Container const& container) noexcept
        : Extent(container.size()), data_(container.data()) {}

    template <size_t ArraySize>
    RPP_HOST_DEVICE constexpr Span(T (&array)[ArraySize]) noexcept
        : Extent(ArraySize), data_(array) {}

    template <size_t ArraySize>
    RPP_HOST_DEVICE constexpr Span(std::array<T, ArraySize> array) noexcept
        : Extent(ArraySize), data_(array.data()) {}

    using Extent::size;

    RPP_HOST_DEVICE RPP_NODISCARD constexpr size_t size_bytes() const noexcept {
        return size() * sizeof(T);
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr bool empty() const noexcept {
        return size() == 0;
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr pointer data() const noexcept {
        return data_;
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator begin() const noexcept {
        return iterator(data());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator end() const noexcept {
        return iterator(data() + size());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr reverse_iterator
    rbegin() const noexcept {
        return reverse_iterator(end());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr reverse_iterator
    rend() const noexcept {
        return reverse_iterator(begin());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr iterator cbegin() const noexcept {
        return const_iterator(data());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr const_iterator
    cend() const noexcept {
        return const_iterator(data() + size());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr const_reverse_iterator
    crbegin() const noexcept {
        return const_iterator(cend());
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr const_reverse_iterator
    crend() const noexcept {
        return const_iterator(cbegin());
    }

    template <typename I>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference
    operator[](I idx) const noexcept {
        return data()[idx];
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference front() const noexcept {
        return *data();
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr reference back() const noexcept {
        return *(data() + size() - 1);
    }

    template <size_t Count>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr Span<T, subspan_extent<Count>>
    first() const noexcept {
        return {data_, std::min(N, Count)};
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Span<T>
    first(size_t count) const noexcept {
        return {data_, std::min(count, N)};
    }

    template <size_t Count>
    RPP_HOST_DEVICE RPP_NODISCARD constexpr Span<T, subspan_extent<Count>>
    last() const noexcept {
        auto offset = (Count >= size()) ? 0 : size() - Count;
        return {data_ + offset, std::min(Count, N)};
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Span<T>
    last(size_t count) const noexcept {
        const auto sz = std::min(count, size());
        const auto offset = size() - count;
        return {data_ + offset, sz};
    }

    template <size_t Offset, size_t Count = dynamic_extent>
    RPP_HOST_DEVICE
        RPP_NODISCARD constexpr Span<T, subspan_extent<Count, Offset>>
        subspan() const noexcept {
        return {data() + Offset, Count};
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Span<T>
    subspan(size_t offset, size_t count = dynamic_extent) const noexcept {
        return {data() + offset,
                count == dynamic_extent ? size() - offset : count};
    }
};

#endif


} // namespace rpp


#endif // RPP_SUPPORT_SPAN_HPP
