#ifndef RPP_DENSE_DETAIL_BASIS_TAG_TRAITS_HPP
#define RPP_DENSE_DETAIL_BASIS_TAG_TRAITS_HPP

#include <type_traits>

namespace rpp::dense::detail {

template<typename T, typename Tag, typename=void>
inline constexpr bool has_matching_basis_tag_v = false;

template<typename T, typename Tag>
inline constexpr bool has_matching_basis_tag_v<T, Tag, std::void_t<typename T::Tag>>
        = std::is_base_of_v<typename T::Tag, Tag>;

} // namespace rpp::dense::detail

#endif // RPP_DENSE_DETAIL_BASIS_TAG_TRAITS_HPP
