#ifndef RPP_SUPPORT_ITERATOR_TRAITS_HPP
#define RPP_SUPPORT_ITERATOR_TRAITS_HPP

#include <iterator>

#include <rpp/config.h>


namespace rpp::traits {

template <typename Iterator, typename = void>
struct IteratorTraits : std::iterator_traits<Iterator> {
};


template <typename It>
using iter_value_t = typename IteratorTraits<It>::value_type;

template <typename It>
using iter_reference_t = typename IteratorTraits<It>::reference;

template <typename It>
using iter_const_reference_t = typename IteratorTraits<It>::const_reference;

template <typename It>
using iter_difference_t = typename IteratorTraits<It>::difference_type;


template <typename It>
inline constexpr bool is_random_access_v = std::is_base_of_v<
    std::random_access_iterator_tag,
    typename IteratorTraits<It>::value_type
>;

template <typename It>
inline constexpr bool is_bidirectional_v = std::is_base_of_v<
    std::random_access_iterator_tag,
    typename IteratorTraits<It>::value_type
>;

template <typename It>
inline constexpr bool is_forward_v = std::is_base_of_v<
    std::forward_iterator_tag,
    typename IteratorTraits<It>::value_type
>;



} /// namespace rpp::traits

#endif //RPP_SUPPORT_ITERATOR_TRAITS_HPP
