#ifndef RPP_BASIS_BASIS_PACK_HPP
#define RPP_BASIS_BASIS_PACK_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/basis/basis_tags.hpp>



namespace rpp::basis {

template <typename... Bases>
struct BasisPack;

namespace detail {

template <typename Tag, typename Basis>
struct BasisHolder : Basis {
    using BasisTag = Tag;
    using BasisType = Basis;

    explicit constexpr BasisHolder(Basis&& basis)
        : Basis(std::move(basis))
    {}
};

template <typename Tag, typename Pack>
constexpr Pack const& get_basis(Pack const& pack RPP_MAYBE_UNUSED) {
    static_assert(
        static_assert_fail<Tag, Pack>,
        "this pack does not contain the specified tag"
        );
    return pack;
}

template <typename Tag, typename Basis>
constexpr Basis const& get_basis(BasisHolder<Tag, Basis> const& h) {
    return static_cast<Basis const&>(h);
}

template <typename Tag, typename First, typename... Rest>
constexpr decltype(auto) get_basis(First const& first, Rest const&... rest) {
    if constexpr (std::is_same_v<Tag, typename First::BasisTag>) {
        return get_basis<Tag>(first);
    } else {
        return get_basis<Tag>(rest...);
    }
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

template <typename Target, typename T, typename... Ts>
constexpr bool has_type() noexcept {
    if constexpr (std::is_same_v<Target, T>) {
        return true;
    } else if constexpr (sizeof...(Ts) > 0) {
        return has_type<Target, Ts...>;
    } else {
        return false;
    }
}

template <typename... SeenTs, typename T, typename... Ts>
constexpr bool check_unique_recurse(std::tuple<SeenTs...> seen, T t, Ts... ts) noexcept {
    ignore_unused(seen, t, ts...);
    if constexpr (has_type<T, SeenTs...>) {
        return false;
    } else {
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
    return !std::is_same_v<T1, T2> && !std::is_same_v<T1, T3> && !std::is_same_v<T2, T3>;
}

template <typename T1, typename T2, typename T3, typename T4>
constexpr bool check_unique(T1 t1, T2 t2, T3 t3, T4 t4) noexcept {
    ignore_unused(t1, t2, t3, t4);
    return check_unique(t1, t2, t3) && !std::is_same_v<T4, T1> && !std::is_same_v<T4, T2> && !std::is_same_v<T4, T3>;
}

template <typename T, typename... Ts>
constexpr bool check_unique(T t, Ts... ts) noexcept {
    ignore_unused(t);
    return check_unique_recurse(std::tuple<T>{}, ts...);
}



} // namespace detail




/**
 * @brief
 *   BasisPack aggregates a variadic list of basis types, each identified by a unique
 *   @c BasisTag. The class stores each basis object (or type) by inheriting from
 *   @c detail::HolderOf<Basis>..., thereby providing compile‑time access to the
 *   underlying bases.
 *
 *   A static assertion (enabled by default) enforces that all tags are distinct at
 *   compile time. The class offers a forwarding constructor that moves the supplied
 *   basis objects into the appropriate holders.
 *
 * @tparam Bases Types of the bases to be packed; each must expose a nested
 *   @c Tag type used for identification.
 *
 * @note When @c RPP_DISABLE_BASIS_PACK_UNIQUENESS_CHECK is defined, the compile‑time
 *   uniqueness check is omitted.
 */
template <typename... Bases>
struct BasisPack : detail::HolderOf<Bases>... {
#ifndef RPP_DISABLE_BASIS_PACK_UNIQUENESS_CHECK
    static_assert(detail::check_unique(typename detail::HolderOf<Bases>::BasisTag {}...),
        "each basis in the pack must have a unique tag"
        );
#endif // RPP_DISABLE_BASIS_PACK_UNIQUENESS_CHECK

    // ReSharper disable once CppNonExplicitConvertingConstructor
    constexpr BasisPack(Bases&&... bases)
        : detail::HolderOf<Bases>{std::move(bases)}...
    {}

};


template <typename TagOrBasis, typename... Bases>
constexpr decltype(auto) get_basis(TagOrBasis const& tag_or_basis, BasisPack<Bases...> const& basis_pack) {
    if constexpr (is_basis_tag_v<TagOrBasis>) {
        return detail::get_basis<TagOrBasis>(static_cast<detail::HolderOf<Bases> const&>(basis_pack)...);
    } else {
        return tag_or_basis;
    }
}

template <typename TagOrBasis, typename Basis>
constexpr decltype(auto) get_basis(TagOrBasis const& tag_or_basis, Basis const& basis) {
    if constexpr (is_basis_tag_v<TagOrBasis>) {
        static_assert(std::is_base_of_v<typename Basis::Tag, TagOrBasis>,
            "the basis provided does not match the provided tag"
            );
        return basis;
    } else {
        return tag_or_basis;
    }
}

template <typename Basis>
constexpr auto in(Basis basis) noexcept -> detail::BasisHolder<InputBasisTag<detail::BasisTagOf<Basis>>, Basis> {
    using Holder = detail::BasisHolder<InputBasisTag<detail::BasisTagOf<Basis>>, Basis>;
    return Holder(std::move(basis));
}

template <typename Basis>
constexpr auto out(Basis basis) noexcept -> detail::BasisHolder<OutputBasisTag<detail::BasisTagOf<Basis>>, Basis> {
    using Holder = detail::BasisHolder<OutputBasisTag<detail::BasisTagOf<Basis>>, Basis>;
    return Holder(std::move(basis));
}

template <size_t Index, typename Basis>
constexpr auto idx(Basis basis) noexcept -> detail::BasisHolder<IndexedBasisTag<Index, detail::BasisTagOf<Basis>>, Basis> {
    using Holder = detail::BasisHolder<IndexedBasisTag<Index, detail::BasisTagOf<Basis>>, Basis>;
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

#endif //RPP_BASIS_BASIS_PACK_HPP
