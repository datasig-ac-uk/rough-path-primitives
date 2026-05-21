#ifndef RPP_SUPPORT_ITERATOR_TRAITS_HPP
#define RPP_SUPPORT_ITERATOR_TRAITS_HPP

#include <iterator>

#include <rpp/config.h>


namespace rpp {
struct HostLocation {};

namespace traits {

template <typename Iterator, typename = void>
struct IteratorTraits : std::iterator_traits<Iterator> {};


template <typename It>
using iter_value_t = typename IteratorTraits<It>::value_type;

template <typename It>
using iter_reference_t = typename IteratorTraits<It>::reference;

template <typename It>
using iter_const_reference_t = typename IteratorTraits<It>::const_reference;

template <typename It>
using iter_difference_t = typename IteratorTraits<It>::difference_type;


template <typename It>
inline constexpr bool is_random_access_v =
    std::is_base_of_v<std::random_access_iterator_tag,
                      typename IteratorTraits<It>::iterator_category>;

template <typename It>
inline constexpr bool is_bidirectional_v =
    std::is_base_of_v<std::random_access_iterator_tag,
                      typename IteratorTraits<It>::iterator_category>;

template <typename It>
inline constexpr bool is_forward_v =
    std::is_base_of_v<std::forward_iterator_tag,
                      typename IteratorTraits<It>::iterator_category>;


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


} // namespace traits

} // namespace rpp

#endif // RPP_SUPPORT_ITERATOR_TRAITS_HPP
