#ifndef RPP_UTILITY_HPP
#define RPP_UTILITY_HPP

#include <cstdint>
#include <type_traits>

#include <rpp/config.h>

namespace rpp {


template <typename... Args>
RPP_HOST_DEVICE
constexpr void ignore_unused(Args&&... arg RPP_MAYBE_UNUSED) noexcept {}

template <typename I>
RPP_HOST_DEVICE constexpr bool is_pow_2(I val) noexcept {
    return (val > 0) && (val & (val - 1)) == I{0};
}

template <typename I, typename E>
RPP_HOST_DEVICE constexpr I const_power(I base, E exp) noexcept {
    I result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result *= base;
        }
        result *= result;
        exp /= 2;
    }
    return result;
}

template <typename T, typename S>
RPP_HOST_DEVICE
constexpr T align_up(T value, S alignment) noexcept {
    return (value + static_cast<T>(alignment) - 1) & ~(static_cast<T>(alignment) - 1);
}


template <unsigned Alignment, typename T>
RPP_HOST_DEVICE inline T* align_up(T* ptr) noexcept {
    if constexpr (std::is_void_v<T>) {
        return ptr;
    } else {
        const auto modifier = static_cast<uintptr_t>(Alignment - 1);
        return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(ptr) + modifier) & ~modifier);
    }
}

}

#endif // RPP_UTILITY_HPP
