#ifndef RPP_SUPPORT_DATA_MAPPING_HPP
#define RPP_SUPPORT_DATA_MAPPING_HPP

#include <tuple>
#include <utility>

#include <rpp/utility.hpp>
#include <rpp/support/error.hpp>


namespace rpp {


template <typename T, typename DataMapper>
constexpr auto
map_data(T&& arg, DataMapper& mapper) noexcept -> typename DataMapper::template Result<std::decay_t<T>> {
    ignore_unused(mapper);
    using RetType = typename DataMapper::template Result<std::decay_t<T>>;
    return RetType{std::forward<T>(arg)};
}

template <template <typename...> class Tuple, typename DataMapper, typename... Ts>
constexpr auto map_data_args(DataMapper& mapper, Ts&&... args) noexcept {
    if constexpr (sizeof...(Ts) == 0) {
        return Result<std::tuple<>, typename DataMapper::Error>(std::make_tuple());
    } else {
        return map_result_tuple(map_to_tuple<Tuple>( [&](auto&& arg) {
            return map_data(std::forward<decltype(arg)>(arg), mapper);
        }, std::forward<Ts>(args)...));
    }
}


} // namespace rpp

#endif //RPP_SUPPORT_DATA_MAPPING_HPP
