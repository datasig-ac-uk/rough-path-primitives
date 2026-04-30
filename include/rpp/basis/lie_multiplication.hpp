#ifndef RPP_BASIS_LIE_MULTIPLICATION_HPP
#define RPP_BASIS_LIE_MULTIPLICATION_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <rpp/basis.hpp>

#include <rpp/architecture.hpp>


namespace rpp {

namespace lie_mul_detail {

template <typename Index>
struct BracketIndex {
    Index left;
    Index right;
};

template <typename Index>
constexpr bool operator==(BracketIndex<Index> const& lhs, BracketIndex<Index> const& rhs) noexcept {
    return lhs.left == rhs.left && lhs.right == rhs.right;
}

template <typename Index>
constexpr bool operator<(BracketIndex<Index> const& lhs, BracketIndex<Index> const& rhs) noexcept {
    return lhs.left < rhs.left || lhs.left == rhs.left && lhs.right < rhs.right;
}



/*
 * This is the FNV1a hashing algorithm.
 */
template <size_t N>
inline constexpr size_t fnv_offset_basis = 0;
template <>
inline constexpr size_t fnv_offset_basis<4> = 0x811c9dc5;
template <>
inline constexpr size_t fnv_offset_basis<8> = 0xcbf29ce484222325;

template <size_t N>
inline constexpr size_t fnv_prime = 0;
template<>
inline constexpr size_t fnv_prime<4> = 0x1000193;
template <>
inline constexpr size_t fnv_prime<8> = 0x100000001b3;

template <typename Index>
struct BracketIndexHash {


    void hash_index(size_t& h, Index index) noexcept {
        #pragma unroll
        for (size_t i=0; i<sizeof(size_t); ++i) {
            h ^= index & size_t{0xFF};
            h *= fnv_prime<sizeof(size_t)>;
            index >>= 8;
        }
    }

    std::size_t operator()(BracketIndex<Index> const& index) const noexcept {
        size_t h = fnv_offset_basis<sizeof(size_t)>;
        hash_index(h, index.left);
        hash_index(h, index.right);
        return h;
    }
};


} // namespace lie_mul_detail


template <typename Architecture=arch::NativeArchitecture>
class LieMultiplicationCache {
    using Degree = typename Architecture::Degree;
    using Index = typename Architecture::Index;
    using LieBasis = LieBasis<Degree, Index>;

    using DataEntry = std::pair<Index, std::make_signed_t<Index>>;
    using DataVec = std::vector<DataEntry>;

    DataVec cache_data_;
    LieBasis basis_;

    using data_iterator = typename DataVec::const_iterator;

public:
    using difference_type = std::ptrdiff_t;

    using BracketIndex = lie_mul_detail::BracketIndex<Index>;

    struct CacheEntry {
        data_iterator begin;
        Index size;
    };

private:

    std::unordered_map<BracketIndex, CacheEntry, lie_mul_detail::BracketIndexHash<Index>> cache_;

    CacheEntry compute_bracket(BracketIndex bracket, Degree degree);




public:

    CacheEntry const& get_bracket(BracketIndex bracket);
    CacheEntry const& get_bracket(Index left, Index right) {
        return get_bracket({left, right});
    }





};

template<typename Architecture>
typename LieMultiplicationCache<Architecture>::CacheEntry LieMultiplicationCache<Architecture>::compute_bracket(
    BracketIndex bracket, Degree degree) {

    if (const auto idx = basis_.find_bracket(bracket.left, bracket.right, degree); idx != 0) {
         cache_data_.emplace_back(bracket.left, idx);
    }

}


template<typename Architecture>
typename LieMultiplicationCache<Architecture>::CacheEntry const & LieMultiplicationCache<Architecture>::get_bracket(
    BracketIndex bracket) {
    static constexpr CacheEntry empty { data_iterator(), 0 };

    if (bracket.left == bracket.right) { return empty; }

    if (bracket.left >= basis_.true_size() || bracket.right >= basis_.true_size()) {
        return empty;
    }

    const auto left_degree = basis_.degree(bracket.left);
    const auto right_degree = basis_.degree(bracket.right);
    const auto degree = left_degree + right_degree;
    if (degree > basis_.depth) { return empty; }

    auto [it, inserted] = cache_.emplace(bracket, empty);
    if (inserted) {
        it->second = compute_bracket(bracket);
    }

    return it->second;
}


} // namespace rpp

#endif //RPP_BASIS_LIE_MULTIPLICATION_HPP
