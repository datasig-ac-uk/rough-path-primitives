#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_ST_MULTIPLY_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_ST_MULTIPLY_HPP

// IWYU pragma: always_keep

#include <algorithm>

#include <rpp/config.h>
#include <rpp/utility.hpp>


#include <rpp/operations/implementation/word_shuffle_product.hpp>

namespace rpp::gpu::block {

template <typename Context,
          typename TensorLhs,
          typename TensorRhs,
          typename Basis>
RPP_DEVICE auto
st_multiply_loop_with_degree(Context const& ctx,
                             typename Context::Index elt_idx,
                             typename Context::Degree elt_degree,
                             Basis const& basis,
                             TensorLhs const& lhs,
                             TensorRhs const& rhs) noexcept {
    const auto rel_idx = elt_idx - basis.start_of_degree(elt_degree);
    return ops::common::word_shuffle_product(
        ctx, rel_idx, elt_degree, lhs, rhs);
}

template <typename Context,
          typename TensorLhs,
          typename TensorRhs,
          typename Basis>
RPP_DEVICE auto st_multiply_loop(Context const& ctx,
                                 typename Context::Index elt_idx,
                                 Basis const& basis,
                                 TensorLhs const& lhs,
                                 TensorRhs const& rhs) noexcept {
    const auto degree = basis.degree(elt_idx);
    return st_multiply_loop_with_degree(ctx, elt_idx, degree, basis, lhs, rhs);
}

} // namespace rpp::gpu::block


#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_ST_MULTIPLY_HPP
