#ifndef RPP_BASIS_HPP
#define RPP_BASIS_HPP

#include <array>
#include <cstdint>
#include <utility>
#include <algorithm>
#include <type_traits>
#include <tuple>

#include <rpp/config.h>
#include <rpp/utility.hpp>


namespace rpp {

namespace detail {

template <typename Degree_, typename Index_>
struct GradedBasis {
    using Degree = Degree_;
    using Index = Index_;

    Degree width;
    Degree depth;
    Index const *degree_begin;

    GradedBasis(Degree width_, Degree depth_, Index const *degree_begin_) noexcept
        : width(width_), depth(depth_), degree_begin(degree_begin_) {
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index size() const noexcept {
        return degree_begin[depth + 1];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index true_size() const noexcept { return size(); }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index start_of_degree(Degree d) const noexcept {
        return degree_begin[d];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index end_of_degree(Degree d) const noexcept {
        return degree_begin[d+1];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index size_of_degree(Degree d) const noexcept {
        return degree_begin[d+1] - degree_begin[d];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Degree degree(Index idx) const noexcept {
        Degree diff = this->depth + 1;
        Degree pos = 0;
        while (diff > 0) {
            const Degree half = diff / 2;
            const Degree new_pos = pos + half;

            if (this->degree_begin[new_pos] <= idx) {
                pos = new_pos + 1;
                diff -= half + 1;
            } else {
                diff = half;
            }
        }
        return pos - 1;
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Degree degree_linear(Index idx) const noexcept {
        Degree result = 0;
        while (result <= depth && degree_begin[result] <= idx) {
            ++result;
        }
        return result - 1;
    }

};


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


template <typename Degree_, typename Index_>
struct TensorBasis : detail::GradedBasis<Degree_, Index_> {
    using Base = detail::GradedBasis<Degree_, Index_>;

    using Degree = typename Base::Degree;
    using Index = typename Base::Index;

    using Base::Base;
    using Base::size;
    using Base::degree;

    RPP_HOST RPP_NODISCARD
    constexpr TensorBasis truncate(Degree new_depth) const noexcept {
        return {
            this->width, std::min(this->depth, new_depth),
            this->degree_begin
        };
    }

    template<typename Array>
    RPP_HOST_DEVICE void unpack_index_to_letters(
        Array& letters,
        Degree degree,
        Index index
    ) const noexcept {
        using Letter = std::remove_reference_t<decltype(letters[0])>;
        for (Degree d = 0; d < degree; ++d) {
            letters[d] = static_cast<Letter>(index % this->width);
            index /= this->width;
        }
    }

    template<typename Array, typename BitMask>
    RPP_HOST_DEVICE
    void pack_masked_index(Array const &letters,
                           Degree degree,
                           BitMask const &bitmask,
                           Degree &lhs_deg,
                           Index &lhs_idx,
                           Degree &rhs_deg,
                           Index &rhs_idx
    ) const noexcept {
        lhs_deg = 0;
        rhs_deg = 0;
        lhs_idx = 0;
        rhs_idx = 0;
        for (; degree >= 0; --degree) {
            bool bit;
            if constexpr (std::is_integral_v<std::remove_cv_t<std::remove_reference_t<BitMask>>>) {
                bit = ((bitmask >> degree) & BitMask{1}) != 0;
            } else {
                bit = static_cast<bool>(bitmask[degree]);
            }
            if (bit) {
                ++lhs_deg;
                lhs_idx = lhs_idx * this->width + letters[degree];
            } else {
                ++rhs_deg;
                rhs_idx = rhs_idx * this->width + letters[degree];
            }
        }
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index reverse_index(Index idx, Degree degree) const noexcept {
        Index result = 0;
        for (Degree d=0; d<degree; ++d) {
            result *= this->width;
            result += idx % this->width;
            idx /= this->width;
        }
        return result;
    }

};

template <typename Degree_, typename Index_, typename Ordering=detail::StandardLieBasisOrder>
struct LieBasis : detail::GradedBasis<Degree_, Index_>, Ordering {
    using Base = detail::GradedBasis<Degree_, Index_>;
    using Degree = typename Base::Degree;
    using Index = typename Base::Index;

    Index const *data;

    using Base::degree;

    explicit constexpr LieBasis(Degree width, Degree depth, Index const *degree_begin,
                                Index const *data_)
        : Base{width, depth, degree_begin}, data{data_} {
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
        return std::tie(data[2*idx], data[2*idx + 1]);
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


using StandardTensorBasis = TensorBasis<std::int32_t, std::ptrdiff_t>;
using StandardLieBasis = LieBasis<std::int32_t, std::ptrdiff_t>;



template <typename Index, typename Width, typename Degree>
RPP_HOST_DEVICE RPP_NODISCARD
constexpr Index tensor_dimension(Width width, Degree degree) noexcept {
    Index result { 1 };
    for (Degree d = 0; d < degree; ++d) {
        result = Index{1} + static_cast<Index>(width) * result;
    }
    return result;
}

template <typename Index, typename Width, typename Degree>
constexpr Index tensor_degree_size(Width width, Degree degree) noexcept {
    return const_power(static_cast<Index>(width), degree);
}

} // namespace rpp


#endif // RPP_BASIS_HPP
