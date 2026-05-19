#ifndef RPP_DENSE_BATCH_HPP
#define RPP_DENSE_BATCH_HPP

#include <type_traits>
#include <utility>

#include <rpp/config.h>
#include <rpp/support/iterator_traits.hpp>
#include <rpp/utility.hpp>

#include <rpp/basis/basis_tags.hpp>
#include <rpp/basis/basis_pack.hpp>

namespace rpp {


template <typename T>
struct ViewMetadata;

namespace layouts {

template <typename Index>
struct StrideLayout {
    Index stride_;

    template <typename It,
              typename = std::enable_if_t<traits::is_random_access_v<It>>>
    RPP_HOST_DEVICE
    auto map(It data, Index index) const noexcept {
        return data + index * stride_;
    }

    template <typename... Its,
              typename = std::enable_if_t<(traits::is_random_access_v<Its> &&
                                           ... && true)>>
    RPP_HOST_DEVICE
    auto map(std::tuple<Its...> data, Index index) const noexcept {
        return map_tuple(data, [&index, this](auto const& it) {
            return it + index * stride_;
        });
    }
};


} // namespace layouts


namespace detail {

template <typename Bases>
struct MetaDataMapper {
    Bases const& bases_;

    template <typename T>
    RPP_HOST_DEVICE
    constexpr auto operator()(T&& arg) const noexcept {
        if constexpr (is_basis_tag_v<T>) {
            return basis::get_basis(arg, bases_);
        } else {
            return std::forward<T>(arg);
        }
    }

};

template <typename Bases, typename Metadata>
RPP_HOST_DEVICE
constexpr auto resolve_metadata(Bases const& bases, Metadata const& meta) noexcept {
    return rpp::map_tuple(meta, MetaDataMapper<Bases>{bases});
}

} // namespace detail


template <typename View,
          typename Layout = layouts::StrideLayout<typename View::Index>,
          typename MetaData = ViewMetadata<View>>
class Batch {
public:
    using Architecture = typename View::Architecture;
    using Data = typename View::Data;
    using Index = typename View::Index;


private:
    std::tuple<Data, Layout, MetaData> data_;

public:
    RPP_HOST_DEVICE
    constexpr decltype(auto) data() const noexcept {
        return std::get<0>(data_);
    }

    RPP_HOST_DEVICE
    constexpr decltype(auto) layout() const noexcept {
        return std::get<1>(data_);
    }

    RPP_HOST_DEVICE
    constexpr decltype(auto) metadata() const noexcept {
        return std::get<2>(data_);
    }

    template <typename BasisPack>
    RPP_HOST_DEVICE
    constexpr View view(BasisPack const& bases, Index idx) const noexcept {
        return View{
            layout().map(data(), idx),
            detail::resolve_metadata(bases, layout().metadata())
        };
    }

};

} // namespace rpp

#endif //RPP_DENSE_BATCH_HPP
