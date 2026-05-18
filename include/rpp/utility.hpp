#ifndef RPP_UTILITY_HPP
#define RPP_UTILITY_HPP

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <tuple>

#include <rpp/config.h>

namespace rpp {
template<typename... Args>
inline constexpr bool static_assert_fail = false;

template<typename... Args>
RPP_HOST_DEVICE
constexpr void ignore_unused(Args &&... arg RPP_MAYBE_UNUSED) noexcept {
}


template <typename T>
RPP_HOST_DEVICE constexpr T* raw_pointer_cast(T* ptr) noexcept { return ptr; }

template<typename I>
RPP_HOST_DEVICE constexpr bool is_pow_2(I val) noexcept {
    return (val > 0) && (val & (val - 1)) == I{0};
}

template<typename I, typename E>
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

template<typename T, typename S>
RPP_HOST_DEVICE
constexpr T align_up(T value, S alignment) noexcept {
    return (value + static_cast<T>(alignment) - 1) & ~(static_cast<T>(alignment) - 1);
}


template<unsigned Alignment, typename T>
RPP_HOST_DEVICE inline T *align_up(T *ptr) noexcept {
    if constexpr (std::is_void_v<T>) {
        return ptr;
    } else {
        const auto modifier = static_cast<uintptr_t>(Alignment - 1);
        return reinterpret_cast<T *>((reinterpret_cast<uintptr_t>(ptr) + modifier) & ~modifier);
    }
}


template<typename I, size_t N>
RPP_HOST_DEVICE constexpr I maximum(const I (&elements)[N]) noexcept {
    static_assert(N > 0, "Maximum of empty array is not defined");
    I max = elements[0];
    for (size_t i = 1; i < N; ++i) {
        max = std::max(elements[i], max);
    }
    return max;
}


template<typename... I>
RPP_HOST_DEVICE constexpr auto maximum(I... elements) noexcept {
    std::common_type_t<I...> vals{elements...};
    return maximum(vals);
}

template<typename I, size_t N>
RPP_HOST_DEVICE constexpr I minimum(const I (&elements)[N]) noexcept {
    static_assert(N > 0, "Minimum of empty array is not defined");
    I max = elements[0];
    for (size_t i = 1; i < N; ++i) {
        max = std::min(elements[i], max);
    }
    return max;
}


template<typename... I>
RPP_HOST_DEVICE constexpr auto minimum(I... elements) noexcept {
    std::common_type_t<I...> vals{elements...};
    return minimum(vals);
}


template <typename I>
RPP_HOST_DEVICE constexpr bool in_range(I arg, I lower, I upper) noexcept {
    return (arg >= lower) && (arg < upper);
}


namespace traits {
namespace detail {
template<typename... Ts>
struct AllSameTypeImpl;

template<typename T>
struct AllSameTypeImpl<T> {
    static constexpr bool value = true;
    using type = T;
};



template<typename T, typename... Ts>
struct AllSameTypeImpl<T, Ts...> {
    using Next = AllSameTypeImpl<Ts...>;
    using type = T;
    static constexpr bool value = Next::value && std::is_same_v<T, typename Next::type>;
};
} // namespace detail

template<typename... Ts>
inline constexpr bool all_same_type_v = detail::AllSameTypeImpl<Ts...>::value;

template<typename... Ts>
using all_same_type_t = std::enable_if_t<detail::AllSameTypeImpl<Ts...>::value, typename detail::AllSameTypeImpl<Ts
    ...>::type>;



template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer_v<T>;


} //namespace traits




namespace detail {

template <template <typename...> class Tuple, typename... Ts, typename Fn, size_t... Is>
constexpr auto map_tuple(Tuple<Ts...> const& arg, Fn&& fn, std::index_sequence<Is...>) noexcept {
    using RetType = Tuple<decltype(fn(std::get<Is>(arg)))...>;
    return RetType(fn(std::get<Is>(arg))...);
}

template <template <typename...> class Tuple, typename... Ts, typename Fn, size_t... Is>
constexpr auto map_tuple(Tuple<Ts...>&& arg, Fn&& fn, std::index_sequence<Is...>) noexcept {
    using RetType = Tuple<decltype(fn(std::get<Is>(std::move(arg))))...>;
    return RetType(fn(std::get<Is>(std::move(arg)))...);
}

} // namespace detail

template <template <typename...> class Tuple, typename... Ts, typename Fn>
constexpr auto map_tuple(Tuple<Ts...> const& arg, Fn&& fn) noexcept {
    return detail::map_tuple(arg, std::forward<Fn>(fn), std::index_sequence_for<Ts...>{});
}
template <template <typename...> class Tuple, typename... Ts, typename Fn>
constexpr auto map_tuple(Tuple<Ts...> && arg, Fn&& fn) noexcept {
    return detail::map_tuple(std::move(arg), std::forward<Fn>(fn), std::index_sequence_for<Ts...>{});
}

template <template <typename...> class Tuple, typename... Ts, typename Fn>
constexpr auto map_to_tuple(Fn&& fn, Ts... args) noexcept {
    using RetType = Tuple<decltype(fn(std::move(args)))...>;
    return RetType(fn(std::move(args))...);
}



}


#endif // RPP_UTILITY_HPP
