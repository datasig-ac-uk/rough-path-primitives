#ifndef RPP_BASIS_TENSOR_BASIS_HPP
#define RPP_BASIS_TENSOR_BASIS_HPP

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/basis/basis_tags.hpp>
#include <rpp/basis/graded_basis.hpp>

namespace rpp::basis {

RPP_MAKE_BASIS_TAG(TensorBasis);

/**
 * @brief Tensor basis representing a graded set of tensor indices.
 *
 * `TensorBasis` provides a lexicographic indexing scheme for tensors of a given
 * depth (`Degree`) and a signed integer index type (`Index`). It inherits from
 * `GradedBasis` and implements utilities for:
 *
 * - Truncating the basis depth (`truncate`).
 * - Unpacking a flat index into an array of letter components
 * (`unpack_index_to_letters`).
 * - Packing a set of letters into a masked index, separating left‑ and
 * right‑hand degree components (`pack_masked_index`).
 * - Reversing a packed index back into a flat representation (`reverse_index`).
 *
 * The class is templated on the numeric degree type and the index type, and
 * leverages host/device macros (`RPP_HOST`, `RPP_HOST_DEVICE`) to support both
 * CPU and GPU code paths. It does not expose raw member properties; interaction
 * should be performed through its member functions.
 *
 */
template <typename Architecture_>
struct TensorBasis : GradedBasis<Architecture_, TensorBasisTag> {
    using Base = GradedBasis<Architecture_, TensorBasisTag>;

    using Degree = typename Base::Degree;
    using Index = typename Base::Index;
    using Architecture = typename Base::Architecture;

    using Base::Base;
    using Base::degree;
    using Base::size;

    RPP_HOST RPP_NODISCARD constexpr TensorBasis
    truncate(Degree new_depth) const noexcept {
        return {
            this->width, std::min(this->depth, new_depth), this->degree_begin};
    }

    template <typename Array>
    RPP_HOST_DEVICE void unpack_index_to_letters(Array& letters,
                                                 Degree degree,
                                                 Index index) const noexcept {
        using Letter = std::remove_reference_t<decltype(letters[0])>;
        for (Degree d = 0; d < degree; ++d) {
            letters[d] = static_cast<Letter>(index % this->width);
            index /= this->width;
        }
    }

    template <typename Array, typename BitMask>
    RPP_HOST_DEVICE void pack_masked_index(Array const& letters,
                                           Degree degree,
                                           BitMask const& bitmask,
                                           Degree& lhs_deg,
                                           Index& lhs_idx,
                                           Degree& rhs_deg,
                                           Index& rhs_idx) const noexcept {
        lhs_deg = 0;
        rhs_deg = 0;
        lhs_idx = 0;
        rhs_idx = 0;
        while (--degree >= 0) {
            if (((bitmask >> degree) & BitMask{1}) != 0) {
                ++lhs_deg;
                lhs_idx = lhs_idx * this->width + letters[degree];
            }
            else {
                ++rhs_deg;
                rhs_idx = rhs_idx * this->width + letters[degree];
            }
        }
    }

    RPP_HOST_DEVICE RPP_NODISCARD constexpr Index
    reverse_index(Index idx, Degree degree) const noexcept {
        Index result = 0;
        for (Degree d = 0; d < degree; ++d) {
            result *= this->width;
            result += idx % this->width;
            idx /= this->width;
        }
        return result;
    }
};

using StandardTensorBasis = TensorBasis<arch::NativeArchitecture>;

template <typename Index, typename Width, typename Degree>
RPP_HOST_DEVICE RPP_NODISCARD constexpr Index
tensor_dimension(Width width, Degree degree) noexcept {
    Index result{1};
    for (Degree d = 0; d < degree; ++d) {
        result = Index{1} + static_cast<Index>(width) * result;
    }
    return result;
}

template <typename Index, typename Width, typename Degree>
constexpr Index tensor_degree_size(Width width, Degree degree) noexcept {
    return const_power(static_cast<Index>(width), degree);
}

} // namespace rpp::basis

#endif // RPP_BASIS_TENSOR_BASIS_HPP
