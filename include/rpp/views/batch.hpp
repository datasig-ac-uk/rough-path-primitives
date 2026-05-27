#ifndef RPP_VIEWS_BATCH_HPP
#define RPP_VIEWS_BATCH_HPP


#include <type_traits>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/basis/basis_pack.hpp>
#include <rpp/basis/basis_tags.hpp>
#include <rpp/support/iterator_traits.hpp>

namespace rpp {


template <typename T, typename = void>
struct ViewMetaData;


template <typename T>
using view_meta_t = typename ViewMetaData<T>::type;

namespace layouts {

struct NoStrideLayout {

    template <typename It,
              typename Index,
              typename = std::enable_if_t<traits::is_random_access_v<It>>>
    RPP_HOST_DEVICE auto map(It data, Index index) const noexcept {
        return data + index;
    }

    template <typename... Its,
              typename Index,
              typename = std::enable_if_t<(traits::is_random_access_v<Its> &&
                                           ... && true)>>
    RPP_HOST_DEVICE auto map(std::tuple<Its...> data,
                             Index index) const noexcept {
        return map_tuple(data,
                         [&index, this](auto const& it) { return it + index; });
    }
};

template <typename Index>
struct StrideLayout {
    Index stride_;

    template <typename It,
              typename = std::enable_if_t<traits::is_random_access_v<It>>>
    RPP_HOST_DEVICE auto map(It data, Index index) const noexcept {
        return data + index * stride_;
    }

    template <typename... Its,
              typename = std::enable_if_t<(traits::is_random_access_v<Its> &&
                                           ... && true)>>
    RPP_HOST_DEVICE auto map(std::tuple<Its...> data,
                             Index index) const noexcept {
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
    RPP_HOST_DEVICE constexpr decltype(auto)
    operator()(T&& arg) const noexcept {
        if constexpr (basis::is_basis_tag_v<std::decay_t<T>>) {
            return basis::get_basis(arg, bases_);
        }
        else {
            return std::forward<T>(arg);
        }
    }
};

template <typename Bases, typename MetaData>
RPP_HOST_DEVICE constexpr auto resolve_metadata(Bases const& bases,
                                                MetaData const& meta) noexcept {
    return rpp::map_tuple(meta, MetaDataMapper<Bases>{bases});
}

} // namespace detail


template <typename View,
          typename Layout = layouts::StrideLayout<typename View::Index>,
          typename MetaData = typename ViewMetaData<View>::type>
class Batch {
public:
    using Architecture = typename View::Architecture;
    using Data = typename View::Data;
    using Index = typename View::Index;


private:
    std::tuple<Data, Layout, MetaData> data_;

public:
    RPP_HOST_DEVICE
    constexpr Batch(Data data, Layout layout, MetaData metadata) noexcept
        : data_(std::move(data), std::move(layout), std::move(metadata)) {}

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

    RPP_HOST_DEVICE
    constexpr View view(Index idx) const noexcept {
        return View{layout().map(data(), idx), metadata()};
    }

    template <typename BasisPack>
    RPP_HOST_DEVICE constexpr View view(BasisPack const& bases,
                                        Index idx) const noexcept {
        return View{layout().map(data(), idx),
                    detail::resolve_metadata(bases, metadata())};
    }
};

template <typename View, typename Layout, typename MetaData, typename Tagger>
RPP_HOST_DEVICE constexpr auto
tag_batch(Batch<View, Layout, MetaData> const& batch,
          Tagger tagger RPP_MAYBE_UNUSED) noexcept;

namespace detail {

template <typename T, typename = void>
struct BasisRefToTag {
    using type = T;
};

template <typename T>
struct BasisRefToTag<const T&, std::void_t<typename T::Tag>> {
    using type = typename T::Tag;
};

template <template <typename...> class Tuple, typename... Args>
struct BasisRefToTag<Tuple<Args...>> {
    using type = Tuple<typename BasisRefToTag<Args>::type...>;
};

} // namespace detail

template <typename T>
struct ViewMetaData<T, std::void_t<typename T::MetaData>> {
    using type = typename detail::BasisRefToTag<typename T::MetaData>::type;
};


namespace batch {

template <typename BasisTagger>
struct MetaDataTagger {

    template <typename T>
    constexpr basis::apply_tagger<BasisTagger, T>
    operator()(T&& arg) const noexcept {
        using Out = basis::apply_tagger<BasisTagger, T>;
        return Out{std::forward<T>(arg)};
    }
};

namespace detail {

template <typename Tagger, typename T>
struct TagMetaDataImpl;

template <typename Tagger, template <typename...> class Tuple, typename... Args>
struct TagMetaDataImpl<Tagger, Tuple<Args...>> {
    using type = Tuple<basis::apply_tagger<Tagger, Args>...>;
};



template <typename Tagger, typename T>
using tagged_metadata_t = typename TagMetaDataImpl<Tagger, T>::type;


template <typename Tagger, typename View, typename Layout, typename MetaData>
using tagged_batch_t = Batch<View, Layout, tagged_metadata_t<Tagger, MetaData>>;

} // namespace detail

template <typename View, typename Layout, typename MetaData, typename Tagger>
RPP_HOST_DEVICE constexpr detail::tagged_batch_t<Tagger, View, Layout, MetaData>
tag_batch(Batch<View, Layout, MetaData> const& batch,
          Tagger&& tagger) noexcept {
    using TaggedBatch = detail::tagged_batch_t<Tagger, View, Layout, MetaData>;

    auto tagged_md = map_tuple(batch.metadata(), std::forward<Tagger>(tagger));
    return TaggedBatch{batch.data(), batch.layout(), std::move(tagged_md)};
}

template <typename View, typename Layout, typename MetaData>
constexpr detail::tagged_batch_t<basis::InputBasisTagger, View, Layout, MetaData>
in(Batch<View, Layout, MetaData> const& batch) noexcept {
    using Tagger = MetaDataTagger<basis::InputBasisTagger>;
    return tag_batch(batch, Tagger{});
}

template <typename View, typename Layout, typename MetaData>
constexpr detail::tagged_batch_t<basis::OutputBasisTagger, View, Layout, MetaData>
out(Batch<View, Layout, MetaData> const& batch) noexcept {
    using Tagger = MetaDataTagger<basis::OutputBasisTagger>;
    return tag_batch(batch, Tagger{});
}

template <size_t Index, typename View, typename Layout, typename MetaData>
constexpr detail::tagged_batch_t<basis::IndexBasisTagger<Index>, View, Layout, MetaData>
idx(Batch<View, Layout, MetaData> const& batch) noexcept {
    using Tagger = MetaDataTagger<basis::IndexBasisTagger<Index>>;
    return tag_batch(batch, Tagger{});
}


} // namespace batch

} // namespace rpp
#endif // RPP_VIEWS_BATCH_HPP
