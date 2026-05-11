#ifndef RPP_BASIS_LIE_BASIS_HPP
#define RPP_BASIS_LIE_BASIS_HPP

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include <rpp/config.h>

#include <rpp/basis/basis_tags.hpp>
#include <rpp/basis/graded_basis.hpp>

namespace rpp {

RPP_MAKE_BASIS_TAG(LieBasis);

namespace detail {

struct StandardLieBasisOrder {

    template <typename Left, typename Right>
    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr bool operator()(Left const& left, Right const& right) const noexcept {
        auto& [a, b] = left;
        auto& [c, d] = right;
        return a < c || (a == c && b < d);
    }
};

} // namespace detail

/**
 * @class LieBasis
 *
 * Represents a Lie algebra basis object used for symbolic and numerical computations.
 * The class encapsulates a set of basis elements and provides operations such as
 * linear combination, product, and commutator evaluation. It supports both dense
 * representations and sparse storage, offering flexible access to basis vectors and
 * their algebraic relations.
 */
template <typename Degree_, typename Index_, typename Ordering=detail::StandardLieBasisOrder>
struct LieBasis : GradedBasis<Degree_, Index_, LieBasisTag>, Ordering {
    using Base = GradedBasis<Degree_, Index_, LieBasisTag>;
    using Degree = typename Base::Degree;
    using Index = typename Base::Index;

    Index const* data;

    using Base::degree;

    explicit constexpr LieBasis(
        Degree width,
        Degree depth,
        Index const* degree_begin,
        Index const* data_
    ) noexcept
        : Base{width, depth, degree_begin}, data{data_}
    {
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr LieBasis truncate(Degree new_depth) const noexcept {
        return LieBasis{
            this->width, std::min(this->depth, new_depth),
            this->degree_begin, this->data
        };
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index size() const noexcept {
        if (this->width <= 1 || this->depth == 0) { return 0; }
        return Base::size() - 1;
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index true_size() const noexcept { return Base::size(); }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr auto operator[](Index idx) const noexcept {
        return std::tie(data[2 * idx], data[2 * idx + 1]);
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index key_to_idx(Index key) const noexcept { return key - 1; }
    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index idx_to_key(Index idx) const noexcept { return idx + 1; }

    template <typename CmpLeft, typename CmpRight>
    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr bool compare(CmpLeft const& left, CmpRight const& right) const noexcept {
        return static_cast<Ordering const&>(*this)(left, right);
    }

    constexpr Index find_bracket(Index left, Index right, Degree degree_hint=0) const noexcept {

        auto deg = degree_hint;
        if (deg == 0) {
            auto left_deg = degree(left);
            auto right_deg = degree(right);
            deg = left_deg + right_deg;

            if (deg > this->depth) { return 0; }
        }

        auto pos = this->start_of_degree(deg);
        const auto end = this->end_of_degree(deg);
        auto diff = end - pos;
        auto needle = std::tie(left, right);

        while (diff > 0) {
            auto half = diff / 2;
            auto test_pos = pos + half;
            auto test = (*this)[test_pos];

            if (test == needle) {
                return test_pos;
            }

            if (compare(test, needle)) {
                pos = test_pos + 1;
                diff -= half + 1;
            } else {
                diff = half;
            }
        }

        if (pos < end && (*this)[pos] == needle) {
            return pos;
        }

        return 0;
    }
};

using StandardLieBasis = LieBasis<std::int32_t, std::ptrdiff_t>;

} // namespace rpp

#endif // RPP_BASIS_LIE_BASIS_HPP
