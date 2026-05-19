#ifndef RPP_BASIS_LIE_BASIS_HPP
#define RPP_BASIS_LIE_BASIS_HPP

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>

#include <rpp/config.h>

#include <rpp/basis/basis_tags.hpp>
#include <rpp/basis/graded_basis.hpp>

namespace rpp::basis {

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
template <typename Architecture_, typename Ordering=detail::StandardLieBasisOrder>
struct LieBasis : GradedBasis<Architecture_, LieBasisTag>, Ordering {
    using Base = GradedBasis<Architecture_, LieBasisTag>;
    using Degree = typename Base::Degree;
    using Index = typename Base::Index;
    using Architecture = typename Base::Architecture;

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

    template <typename DataMapper>
    RPP_NODISCARD
    friend typename DataMapper::template Result<LieBasis<typename DataMapper::Architecture, Ordering>>
    map_data(LieBasis const& basis, DataMapper& mapper) noexcept {
        if constexpr (std::is_same_v<Architecture_, typename DataMapper::Architecture>) {
            return basis;
        } else {
            using TgtIndex = typename DataMapper::Architecture::Index;

            auto mapped_db = mapper.template copy_n<TgtIndex>(basis.degree_begin, basis.depth + 2);
            if (!mapped_db) { return std::move(mapped_db).error(); }

            auto mapped_data = mapper.template copy_n<TgtIndex>(basis.data, mapped_db.size());
            if (!mapped_data) { return std::move(mapped_data).error(); }

            return LieBasis<typename DataMapper::Architecture, Ordering>{
                mapped_db.width, mapped_db.depth, mapped_db.value(), mapped_data.value()
            };
        }
    }
};

using StandardLieBasis = LieBasis<arch::NativeArchitecture>;

} // namespace rpp::basis


#endif // RPP_BASIS_LIE_BASIS_HPP
