#ifndef RPP_BASIS_TENSOR_BASIS_HPP
#define RPP_BASIS_TENSOR_BASIS_HPP

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/basis/basis_tags.hpp>
#include <rpp/basis/graded_basis.hpp>

namespace rpp {

RPP_MAKE_BASIS_TAG(TensorBasis);

template <typename Degree_, typename Index_>
struct TensorBasis : GradedBasis<Degree_, Index_, TensorBasisTag> {
    using Base = GradedBasis<Degree_, Index_, TensorBasisTag>;

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

    template <typename Array>
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

    template <typename Array, typename BitMask>
    RPP_HOST_DEVICE
    void pack_masked_index(
        Array const& letters,
        Degree degree,
        BitMask const& bitmask,
        Degree& lhs_deg,
        Index& lhs_idx,
        Degree& rhs_deg,
        Index& rhs_idx
    ) const noexcept {
        lhs_deg = 0;
        rhs_deg = 0;
        lhs_idx = 0;
        rhs_idx = 0;
        while (--degree >= 0) {
            if (((bitmask >> degree) & BitMask{1}) != 0) {
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
        for (Degree d = 0; d < degree; ++d) {
            result *= this->width;
            result += idx % this->width;
            idx /= this->width;
        }
        return result;
    }
};

using StandardTensorBasis = TensorBasis<std::int32_t, std::ptrdiff_t>;

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

#endif // RPP_BASIS_TENSOR_BASIS_HPP
