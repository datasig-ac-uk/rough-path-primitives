#ifndef RPP_SUPPORT_DATA_MAPPING_HPP
#define RPP_SUPPORT_DATA_MAPPING_HPP

#include <tuple>
#include <utility>

#include <rpp/utility.hpp>
#include <rpp/support/error.hpp>


namespace rpp {


template <typename T, typename DataMapper>
constexpr
typename DataMapper::template Result<T>
map_data(T const& arg, DataMapper& mapper) noexcept {
    return mapper.template map<T>(arg);
}

template <template <typename...> class Tuple, typename DataMapper, typename... Ts>
constexpr auto map_data_args(DataMapper& mapper, Ts const&... args) noexcept {
    return map_result_tuple(map_to_tuple<Tuple>( [&](auto const& arg) {
        return map_data(arg, mapper);
    }, args...));
}


} // namespace rpp

#endif //RPP_SUPPORT_DATA_MAPPING_HPP
