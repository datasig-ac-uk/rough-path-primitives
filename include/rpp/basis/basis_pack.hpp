#ifndef RPP_BASIS_BASIS_PACK_HPP
#define RPP_BASIS_BASIS_PACK_HPP

#include <tuple>
#include <utility>

#include <rpp/basis/basis_tags.hpp>
#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/support/data_mapping.hpp>

namespace rpp::basis {

template <typename... Bases>
struct BasisPack;


namespace detail {


template <typename Tag, typename Basis>
struct BasisHolder : Basis {
    using BasisTag = Tag;
    using BasisType = Basis;

    explicit constexpr BasisHolder(Basis&& basis) : Basis(std::move(basis)) {}
};


template <typename Mapper, typename Tag, typename Basis>
constexpr typename Mapper::template Result<BasisHolder<Tag, traits::data_mapped_t<Mapper, Basis>>>
map_data(Mapper& mapper, BasisHolder<Tag, Basis> const& h) noexcept {
    using Holder = BasisHolder<Tag, traits::data_mapped_t<Mapper, Basis>>;
    auto mapped_basis = map_data(mapper, static_cast<Basis const&>(h));
    if (!mapped_basis) { return std::move(mapped_basis).error(); }
    return Holder{std::move(mapped_basis).value()};
}


template <typename T>
struct HolderOfImpl {
    using type = BasisHolder<typename T::Tag, T>;
};

template <typename T, typename B>
struct HolderOfImpl<BasisHolder<T, B>> {
    using type = BasisHolder<T, B>;
};

template <typename Basis>
using HolderOf = typename HolderOfImpl<Basis>::type;

template <typename T>
struct BasisTagOfImpl {
    using type = typename T::Tag;
};

template <typename Tag, typename Basis>
struct BasisTagOfImpl<BasisHolder<Tag, Basis>> {
    using type = Tag;
};

template <typename T>
using BasisTagOf = typename BasisTagOfImpl<T>::type;


template <typename... B>
using BasisPackBase = std::tuple<HolderOf<B>...>;


template <typename Tag, typename Pack>
RPP_HOST_DEVICE
constexpr Pack const&
get_basis(Pack const& pack RPP_MAYBE_UNUSED,
          std::index_sequence<> seq RPP_MAYBE_UNUSED = {}) {
    static_assert(static_assert_fail<Tag, Pack>,
                  "this pack does not contain the specified tag");
    return pack;
}

template <typename Tag, typename Basis>
RPP_HOST_DEVICE
constexpr Basis const& get_basis(BasisHolder<Tag, Basis> const& h) {
    return static_cast<Basis const&>(h);
}

template <typename Tag, typename Pack, size_t I, size_t... Is>
RPP_HOST_DEVICE
constexpr decltype(auto)
get_basis(Pack const& pack,
          std::index_sequence<I, Is...> seq RPP_MAYBE_UNUSED) noexcept {
    using EltType = std::tuple_element_t<I, Pack>;

    if constexpr (std::is_same_v<Tag, typename EltType::BasisTag>) {
        return get_basis(std::get<I>(pack));
    }
    else {
        return get_basis<Tag>(pack, std::index_sequence<Is...>{});
    }
}


#ifndef RPP_DISABLE_BASIS_PACK_UNIQUENESS_CHECK
template <typename Target, typename T, typename... Ts>
constexpr bool has_type() noexcept {
    if constexpr (std::is_same_v<Target, T>) {
        return true;
    }
    else if constexpr (sizeof...(Ts) > 0) {
        return has_type<Target, Ts...>;
    }
    else {
        return false;
    }
}

template <typename... SeenTs, typename T, typename... Ts>
constexpr bool
check_unique_recurse(std::tuple<SeenTs...> seen, T t, Ts... ts) noexcept {
    ignore_unused(seen, t, ts...);
    if constexpr (has_type<T, SeenTs...>) {
        return false;
    }
    else {
        return check_unique_recurse(std::tuple<SeenTs..., T>{}, ts...);
    }
}

constexpr bool check_unique() noexcept { return true; }

template <typename T>
constexpr bool check_unique(T t) noexcept {
    ignore_unused(t);
    return true;
}

template <typename T1, typename T2>
constexpr bool check_unique(T1 t1, T2 t2) noexcept {
    ignore_unused(t1, t2);
    return !std::is_same_v<T1, T2>;
}

template <typename T1, typename T2, typename T3>
constexpr bool check_unique(T1 t1, T2 t2, T3 t3) noexcept {
    ignore_unused(t1, t2, t3);
    return !std::is_same_v<T1, T2> && !std::is_same_v<T1, T3> &&
        !std::is_same_v<T2, T3>;
}

template <typename T1, typename T2, typename T3, typename T4>
constexpr bool check_unique(T1 t1, T2 t2, T3 t3, T4 t4) noexcept {
    ignore_unused(t1, t2, t3, t4);
    return check_unique(t1, t2, t3) && !std::is_same_v<T4, T1> &&
        !std::is_same_v<T4, T2> && !std::is_same_v<T4, T3>;
}

template <typename T, typename... Ts>
constexpr bool check_unique(T t, Ts... ts) noexcept {
    ignore_unused(t);
    return check_unique_recurse(std::tuple<T>{}, ts...);
}

#endif

} // namespace detail


/**
 * @brief
 *   BasisPack aggregates a variadic list of basis types, each identified by a
 * unique
 *   @c BasisTag. The class stores each basis object (or type) by inheriting
 * from
 *   @c detail::HolderOf<Basis>..., thereby providing compile‑time access to the
 *   underlying bases.
 *
 *   A static assertion (enabled by default) enforces that all tags are distinct
 * at compile time. The class offers a forwarding constructor that moves the
 * supplied basis objects into the appropriate holders.
 *
 * @tparam Bases Types of the bases to be packed; each must expose a nested
 *   @c Tag type used for identification.
 *
 * @note When @c RPP_DISABLE_BASIS_PACK_UNIQUENESS_CHECK is defined, the
 * compile‑time uniqueness check is omitted.
 */
template <typename... Bases>
struct BasisPack : public detail::BasisPackBase<Bases...> {
    using Base = detail::BasisPackBase<Bases...>;
#ifndef RPP_DISABLE_BASIS_PACK_UNIQUENESS_CHECK
    static_assert(
        detail::check_unique(typename detail::HolderOf<Bases>::BasisTag{}...),
        "each basis in the pack must have a unique tag");
#endif // RPP_DISABLE_BASIS_PACK_UNIQUENESS_CHECK
private:

    RPP_HOST_DEVICE
    explicit constexpr BasisPack(Base&& base) : Base(std::move(base))
    {}

public:
    RPP_HOST_DEVICE
    constexpr BasisPack(Bases&&... bases)
        : Base(std::move(bases)...)
    {}

    template <size_t I>
    RPP_HOST_DEVICE
    constexpr decltype(auto) get() const noexcept {
        return std::get<I>(static_cast<Base const&>(*this));
    }

    template <size_t I>
    RPP_HOST_DEVICE
    friend constexpr decltype(auto) get(BasisPack const& pack) noexcept {
        return std::get<I>(static_cast<Base const &>(pack));
    }

    template <size_t I>
    RPP_HOST_DEVICE
    friend constexpr decltype(auto) get(BasisPack&& pack) noexcept {
        return std::get<I>(static_cast<Base&&>(pack));
    }


    template <typename DataMapper>
    RPP_HOST
    friend constexpr auto map_data(DataMapper& mapper, BasisPack const& pack) noexcept {
        using ReturnType = BasisPack<traits::data_mapped_t<DataMapper, Bases>...>;
        using Result = typename DataMapper::template Result<ReturnType>;
        auto mapped_base = map_data(mapper, static_cast<Base const&>(pack));
        if (!mapped_base) {
            return std::move(mapped_base).error();
        }

        return Result{std::move(mapped_base).value()};
    }
};


template <typename TagOrBasis, typename... Bases>
RPP_HOST_DEVICE
constexpr decltype(auto) get_basis(TagOrBasis const& tag_or_basis,
                                   BasisPack<Bases...> const& basis_pack) {
    if constexpr (is_basis_tag_v<TagOrBasis>) {
        return detail::get_basis<TagOrBasis>(
            basis_pack,
            std::make_index_sequence<std::tuple_size_v<BasisPack<Bases...>>>{});
    }
    else {
        return tag_or_basis;
    }
}

template <typename TagOrBasis, typename Basis>
RPP_HOST_DEVICE
constexpr decltype(auto) get_basis(TagOrBasis const& tag_or_basis,
                                   Basis const& basis) {
    if constexpr (is_basis_tag_v<TagOrBasis>) {
        static_assert(std::is_base_of_v<typename Basis::Tag, TagOrBasis>,
                      "the basis provided does not match the provided tag");
        return basis;
    }
    else {
        return tag_or_basis;
    }
}

template <size_t I, typename... Bases>
RPP_HOST_DEVICE
constexpr decltype(auto) get_basis(BasisPack<Bases...> const& pack) noexcept {
    return detail::get_basis(pack.template get<I>());
}

template <typename Basis>
constexpr auto in(Basis basis) noexcept
    -> detail::BasisHolder<InputBasisTag<detail::BasisTagOf<Basis>>, Basis> {
    using Holder =
        detail::BasisHolder<InputBasisTag<detail::BasisTagOf<Basis>>, Basis>;
    return Holder(std::move(basis));
}

template <typename Basis>
constexpr auto out(Basis basis) noexcept
    -> detail::BasisHolder<OutputBasisTag<detail::BasisTagOf<Basis>>, Basis> {
    using Holder =
        detail::BasisHolder<OutputBasisTag<detail::BasisTagOf<Basis>>, Basis>;
    return Holder(std::move(basis));
}

template <size_t Index, typename Basis>
constexpr auto idx(Basis basis) noexcept
    -> detail::BasisHolder<IndexedBasisTag<Index, detail::BasisTagOf<Basis>>,
                           Basis> {
    using Holder =
        detail::BasisHolder<IndexedBasisTag<Index, detail::BasisTagOf<Basis>>,
                            Basis>;
    return Holder(std::move(basis));
}

/**
 * @brief Constructs a @c BasisPack from a variadic list of basis objects.
 *
 * @tparam Bases Types of the basis objects; each must provide a nested
 *   @c Tag type that uniquely identifies the basis.
 *
 * @param bases A pack of basis objects of types @c Bases... to be packed.
 *   Each object is moved into the appropriate holder within the resulting
 *   @c BasisPack.
 *
 * @return A @c BasisPack<Bases...> containing the supplied basis objects.
 */
template <typename... Bases>
constexpr BasisPack<Bases...> make_basis_pack(Bases... bases) {
    return {std::move(bases)...};
}

} // namespace rpp::basis



template <typename... Bases>
struct std::tuple_size<rpp::basis::BasisPack<Bases...>> : integral_constant<size_t, sizeof...(Bases)> {}; // NOLINT(*-dcl58-cpp)

template <size_t I, typename... Bases>
struct std::tuple_element<I, rpp::basis::BasisPack<Bases...>> // NOLINT(*-dcl58-cpp)
    : std::tuple_element<I, rpp::basis::detail::BasisPackBase<Bases...>> {};




#endif // RPP_BASIS_BASIS_PACK_HPP
