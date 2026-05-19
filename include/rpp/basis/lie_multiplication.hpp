#ifndef RPP_BASIS_LIE_MULTIPLICATION_HPP
#define RPP_BASIS_LIE_MULTIPLICATION_HPP

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <map>

#include <rpp/architecture.hpp>
#include <rpp/basis/lie_basis.hpp>


namespace rpp::basis {
namespace lie_mul_detail {
template<typename Index>
struct BracketIndex {
    Index left;
    Index right;
};

template<typename Index>
constexpr bool operator==(BracketIndex<Index> const &lhs, BracketIndex<Index> const &rhs) noexcept {
    return lhs.left == rhs.left && lhs.right == rhs.right;
}

template<typename Index>
constexpr bool operator<(BracketIndex<Index> const &lhs, BracketIndex<Index> const &rhs) noexcept {
    return lhs.left < rhs.left || lhs.left == rhs.left && lhs.right < rhs.right;
}


/*
 * This is the FNV1a hashing algorithm.
 */
template<size_t N>
inline constexpr size_t fnv_offset_basis = 0;
template<>
inline constexpr size_t fnv_offset_basis<4> = 0x811c9dc5;
template<>
inline constexpr size_t fnv_offset_basis<8> = 0xcbf29ce484222325;

template<size_t N>
inline constexpr size_t fnv_prime = 0;
template<>
inline constexpr size_t fnv_prime<4> = 0x1000193;
template<>
inline constexpr size_t fnv_prime<8> = 0x100000001b3;

template<typename Index>
struct BracketIndexHash {
    static void hash_index(size_t &h, Index index) noexcept {
#pragma unroll
        for (size_t i = 0; i < sizeof(size_t); ++i) {
            h ^= index & size_t{0xFF};
            h *= fnv_prime<sizeof(size_t)>;
            index >>= 8;
        }
    }

    std::size_t operator()(BracketIndex<Index> const &index) const noexcept {
        size_t h = fnv_offset_basis<sizeof(size_t)>;
        hash_index(h, index.left);
        hash_index(h, index.right);
        return h;
    }
};
} // namespace lie_mul_detail


template<typename Architecture=arch::NativeArchitecture>
class LieMultiplicationCache {
    using Degree = typename Architecture::Degree;
    using Index = typename Architecture::Index;
    using Basis = LieBasis<Architecture>;

    using CacheInteger = std::make_signed_t<Index>;
    using DataEntry = std::pair<Index, CacheInteger>;

    Basis basis_;

public:
    using difference_type = std::ptrdiff_t;

    using BracketIndex = lie_mul_detail::BracketIndex<Index>;
    using CacheEntry = std::vector<DataEntry>;

    explicit LieMultiplicationCache(Basis basis) : basis_(basis) {};

private:
    std::unordered_map<BracketIndex, CacheEntry, lie_mul_detail::BracketIndexHash<Index> > cache_;


    static auto get_scalar(CacheEntry &entry, Index index) {
        auto it = std::lower_bound(entry.begin(), entry.end(), index,
                                   [](DataEntry const &l, Index r) noexcept { return l.first < r; });
        if (it == entry.end() || it->first != index) {
            it = entry.insert(it, {index, 0});
        }
        return it;
    }

    void compute_bracket_half(CacheEntry &entry, BracketIndex outer, Index inner_right, CacheInteger sign);
    void compute_bracket(CacheEntry &entry, BracketIndex bracket, Degree degree);
    void append_scaled(CacheEntry &entry, CacheEntry const &other, CacheInteger sign);
public:
    CacheEntry const &get_bracket(BracketIndex bracket);

    CacheEntry const &get_bracket(Index left, Index right) {
        return get_bracket({left, right});
    }
};

template<typename Architecture>
void LieMultiplicationCache<Architecture>::compute_bracket_half(CacheEntry &entry, BracketIndex outer,
    Index inner_right, CacheInteger sign) {
    constexpr CacheInteger zero { 0 };

    auto outer_product = get_bracket(outer);

    for (auto const& [outer_idx, outer_val] : outer_product) {
        BracketIndex inner {outer_idx, inner_right};
        auto inner_product = get_bracket(inner);

        for (auto const& [inner_idx, inner_val] : inner_product) {
            auto it = get_scalar(entry, inner_idx);
            if ((it->second += sign * outer_val * inner_val) == zero) {
                entry.erase(it);
            }
        }
    }

}


template<typename Architecture>
void LieMultiplicationCache<Architecture>::compute_bracket(
    CacheEntry &entry, BracketIndex bracket, Degree degree) {
    if (const auto idx = basis_.find_bracket(bracket.left, bracket.right, degree); idx != 0) {
        entry.emplace_back(idx, 1);
        return;
    }

    if (const auto idx = basis_.find_bracket(bracket.right, bracket.left, degree); idx != 0) {
        entry.emplace_back(idx, -1);
        return;
    }

    if (bracket.right < bracket.left) {
        auto const& reverse = get_bracket({bracket.right, bracket.left});
        entry.reserve(reverse.size());
        for (auto const& [idx, val] : reverse) {
            entry.emplace_back(idx, -val);
        }
        return;
    }

    auto [lparent, rparent] = basis_[bracket.right];

    if (lparent > 0 && lparent != bracket.left) {
        BracketIndex outer {bracket.left, lparent};
        compute_bracket_half(entry, outer, rparent, 1);
    }

    if (rparent != bracket.left) {
        BracketIndex outer {bracket.left, rparent};
        compute_bracket_half(entry, outer, lparent, -1);
    }

}


template<typename Architecture>
typename LieMultiplicationCache<Architecture>::CacheEntry const &LieMultiplicationCache<Architecture>::get_bracket(
    BracketIndex bracket) {
    static const CacheEntry empty{};

    if (bracket.left <= 0 || bracket.right <= 0 || bracket.left == bracket.right) { return empty; }

    if (bracket.left >= basis_.true_size() || bracket.right >= basis_.true_size()) {
        return empty;
    }

    const auto left_degree = basis_.degree(bracket.left);
    const auto right_degree = basis_.degree(bracket.right);
    const auto degree = left_degree + right_degree;
    if (degree > basis_.depth) { return empty; }

    auto [it, inserted] = cache_.emplace(bracket, empty);
    if (inserted) {
        compute_bracket(it->second, bracket, degree);
    }

    return it->second;
}
} // namespace rpp::basis

namespace rpp {

template<typename Architecture=arch::NativeArchitecture>
using LieMultiplicationCache = basis::LieMultiplicationCache<Architecture>;

namespace lie_mul_detail {
template<typename Index>
using BracketIndex = basis::lie_mul_detail::BracketIndex<Index>;

template<typename Index>
using BracketIndexHash = basis::lie_mul_detail::BracketIndexHash<Index>;
} // namespace lie_mul_detail

} // namespace rpp

#endif //RPP_BASIS_LIE_MULTIPLICATION_HPP
