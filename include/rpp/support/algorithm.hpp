#ifndef RPP_SUPPORT_ALGORITHM_HPP
#define RPP_SUPPORT_ALGORITHM_HPP

#include <functional>

#include <rpp/config.h>

namespace rpp::algo {


template <typename It, typename I, typename J, typename Compare>
RPP_HOST_DEVICE constexpr I index_lower_bound(
    It const& it, I pos, I end, J const& val, Compare comp) noexcept {
    I count = end - pos;

    while (count > 0) {
        I half = count / 2;

        if (comp(it[pos + half], val)) {
            pos = pos + half + 1;
            count -= half + 1;
        }
        else {
            count = half;
        }
    }

    return pos;
}


template <typename It, typename I, typename J>
RPP_HOST_DEVICE constexpr I
index_lower_bound(It const& it, I pos, I end, J const& val) noexcept {
    return index_lower_bound(it, pos, end, val, std::less<>{});
}

template <typename It, typename I, typename J, typename Compare>
RPP_HOST_DEVICE constexpr I index_upper_bound(
    It const& it, I pos, I end, J const& val, Compare comp) noexcept {
    I count = end - pos;

    while (count > 0) {
        I half = count / 2;

        if (!comp(val, it[pos + half])) {
            pos = pos + half + 1;
            count -= half + 1;
        }
        else {
            count = half;
        }
    }

    return pos;
}


template <typename It, typename I, typename J>
RPP_HOST_DEVICE constexpr I
index_upper_bound(It const& it, I pos, I end, J const& val) noexcept {
    return index_upper_bound(it, pos, end, val, std::less<>{});
}


} // namespace rpp::algo


#endif // RPP_SUPPORT_ALGORITHM_HPP
