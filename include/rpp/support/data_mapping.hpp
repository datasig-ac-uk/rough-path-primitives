#ifndef RPP_SUPPORT_DATA_MAPPING_HPP
#define RPP_SUPPORT_DATA_MAPPING_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include <rpp/support/error.hpp>
#include <rpp/support/iterator_traits.hpp>
#include <rpp/utility.hpp>



namespace rpp {


namespace traits {
template <typename DataMapper, typename T>
using data_map_result_t = typename DataMapper::template Result<T>;

template <typename DataMapper, typename T>
using map_result_t = decltype(map_data(std::declval<DataMapper&>(),
                                       std::declval<T const&>()));

template <typename DataMapper, typename T>
using mapped_value_t = typename map_result_t<DataMapper, T>::value_type;

template <typename Mapper, typename It, typename Value = iter_value_t<It>>
using data_map_value_t =
    std::conditional_t<Mapper::template has_data<It>() &&
                           std::is_same_v<Value, iter_value_t<It>>,
                       It,
                       typename Mapper::template Ptr<Value>>;

template <typename DataMapper, typename It>
using data_map_index_t = data_map_value_t<DataMapper, It, typename DataMapper::Index>;

template <typename DataMapper, typename It>
using data_map_degree_t = data_map_value_t<DataMapper, It, typename DataMapper::Degree>;


} // namespace traits


template <typename DataMapper, typename T>
constexpr auto map_data(DataMapper& mapper, T const& arg) noexcept ->
    typename DataMapper::template Result<T> {
    ignore_unused(mapper);
    using RetType = typename DataMapper::template Result<T>;
    return RetType{T{arg}};
}


template <typename DataMapper, typename... Ts>
constexpr auto map_data(DataMapper& mapper, std::tuple<Ts...> const& arg) noexcept
    -> traits::data_map_result_t<DataMapper,
                                 std::tuple<traits::mapped_value_t<DataMapper, Ts>...>>
{
    auto mapped = map_tuple(arg, [&](auto const& tv) { return map_data(mapper, tv); });
    return map_result_tuple(std::move(mapped));
}

template <template <typename...> class Tuple,
          typename DataMapper,
          typename... Ts>
constexpr auto map_data_args(DataMapper& mapper, Ts const&... args) noexcept {
    if constexpr (sizeof...(Ts) == 0) {
        return Result<std::tuple<>, typename DataMapper::Error>(
            std::make_tuple());
    }
    else {
        return map_result_tuple(map_to_tuple<Tuple>(
            [&](auto const& arg) {
                return map_data(mapper, std::forward<decltype(arg)>(arg));
            },
            args...));
    }
}

template <typename DataMapper, typename It>
constexpr auto map_value_range(DataMapper& mapper, It begin, It end) noexcept {
    if constexpr (!mapper.template has_data<It>()) {
        using Traits = traits::IteratorTraits<It>;
        return mapper.template copy_n<typename Traits::value_type>(std::move(begin),
                                          std::distance(begin, end));
    }
    else {
        return typename DataMapper::template Result<It>(begin);
    }
}

template <typename DataMapper, typename It, typename Size>
constexpr auto
map_value_range(DataMapper& mapper, It begin, Size size) noexcept {
    if constexpr (!mapper.template has_data<It>()) {
        using Traits = traits::IteratorTraits<It>;
        return mapper.template copy_n<typename Traits::value_type>(std::move(begin), size);
    }
    else {
        return typename DataMapper::template Result<It>(begin);
    }
}

template <typename DataMapper, typename It>
constexpr auto map_index_range(DataMapper& mapper, It begin, It end) noexcept {
    using Traits = traits::IteratorTraits<It>;
    using Index = typename DataMapper::Index;
    using Value = typename Traits::value_type;


    if constexpr (!mapper.template has_data<It>() ||
                  !std::is_same_v<Value, Index>) {
        return mapper.template copy_n<Index>(std::move(begin), std::move(end));
    }
    else {
        return typename DataMapper::template Result<It>(begin);
    }
}

template <typename DataMapper, typename It, typename Size>
constexpr auto
map_index_range(DataMapper& mapper, It begin, Size size) noexcept {
    using Traits = traits::IteratorTraits<It>;
    using Index = typename DataMapper::Index;
    using Value = typename Traits::value_type;

    if constexpr (!mapper.template has_data<It>() ||
                  !std::is_same_v<Value, Index>) {
        return mapper.template copy_n<Index>(std::move(begin), size);
    }
    else {
        return typename DataMapper::template Result<It>(begin);
    }
}

template <typename DataMapper, typename It>
constexpr auto map_degree_range(DataMapper& mapper, It begin, It end) noexcept {
    using Traits = traits::IteratorTraits<It>;
    using Degree = typename DataMapper::Degree;
    using Value = typename Traits::value_type;

    if constexpr (!mapper.template has_data<It>() ||
                  !std::is_same_v<Value, Degree>) {
        return mapper.template copy_n<Degree>(std::move(begin), std::move(end));
    }
    else {
        return typename DataMapper::template Result<It>(begin);
    }
}

template <typename DataMapper, typename It, typename Size>
constexpr auto
map_degree_range(DataMapper& mapper, It begin, Size size) noexcept {
    using Traits = traits::IteratorTraits<It>;
    using Degree = typename DataMapper::Degree;
    using Value = typename Traits::value_type;

    if constexpr (!mapper.template has_data<It>() ||
                  !std::is_same_v<Value, Degree>) {
        return mapper.template copy_n<Degree>(std::move(begin), size);
    }
    else {
        return typename DataMapper::template Result<It>(begin);
    }
}


} // namespace rpp

#endif // RPP_SUPPORT_DATA_MAPPING_HPP
