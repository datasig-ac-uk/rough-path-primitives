#ifndef INCLUDE_RPP_OPERATIONS_IMPLEMENTATION_WORD_SHUFFLE_PRODUCT_HPP
#define INCLUDE_RPP_OPERATIONS_IMPLEMENTATION_WORD_SHUFFLE_PRODUCT_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/views/dense_tensor_view.hpp>


namespace rpp::ops::common {

namespace detail {

template <typename Tensor>
struct DefaultGetter {
    using Index = typename Tensor::Index;
    using Degree = typename Tensor::Degree;

    RPP_HOST_DEVICE RPP_FORCEINLINE bool
    has_degree(Tensor const& instance, Degree degree) const noexcept {
        ignore_unused(this);
        return instance.has_degree(degree);
    }

    RPP_HOST_DEVICE RPP_FORCEINLINE decltype(auto) operator()(
        Tensor const& instance, Degree degree, Index index) const noexcept {
        return instance.degree_view(degree)[index];
    }
};

} // namespace detail

template <typename Context,
          typename TensorLhs,
          typename TensorRhs,
          typename LhsGetter = detail::DefaultGetter<TensorLhs>,
          typename RhsGetter = detail::DefaultGetter<TensorRhs>>
RPP_HOST_DEVICE RPP_FORCEINLINE typename Context::Accum
word_shuffle_product(Context const& ctx RPP_MAYBE_UNUSED,
                     typename Context::Index elt_rel_index,
                     typename Context::Degree elt_degree,
                     TensorLhs const& lhs,
                     TensorRhs const& rhs,
                     LhsGetter&& lhs_getter = LhsGetter{},
                     RhsGetter&& rhs_getter = RhsGetter{}) noexcept {
    using Strategy = typename Context::Strategy;
    using Index = typename Context::Index;
    using Degree = typename Context::Degree;
    using Accum = typename Context::Accum;
    using Letter = typename Context::Letter;
    using Bitmask = typename Context::Bitmask;

    const auto& basis = lhs.basis();

    Letter letters[Strategy::Architecture::max_depth];

    basis.unpack_index_to_letters(letters, elt_degree, elt_rel_index);

    const auto right_min_degree =
        std::max<Degree>(elt_degree - lhs.max_degree(), rhs.min_degree());
    const auto right_max_degree =
        std::min<Degree>(elt_degree - lhs.min_degree(), rhs.max_degree());

    Accum acc{0};

    for (Bitmask mask{0}; mask < (Bitmask{1} << elt_degree); ++mask) {
        Index left_idx, right_idx;
        Degree left_deg, right_deg;
        basis.pack_masked_index(letters,
                                elt_degree,
                                mask,
                                left_deg,
                                left_idx,
                                right_deg,
                                right_idx);

        if (lhs_getter.has_degree(lhs, left_deg) &&
            rhs_getter.has_degree(rhs, right_deg)) {
            acc += lhs_getter(lhs, left_deg, left_idx) *
                rhs_getter(rhs, right_deg, right_idx);
        }
    }

    return acc;
}

} // namespace rpp::ops::common


#endif // INCLUDE_RPP_OPERATIONS_IMPLEMENTATION_WORD_SHUFFLE_PRODUCT_HPP
