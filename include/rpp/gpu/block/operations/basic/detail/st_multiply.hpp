#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_ST_MULTIPLY_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_ST_MULTIPLY_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/utility.hpp>


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
    using Strategy = typename Context::Strategy;
    using Index = typename Context::Index;
    using Degree = typename Context::Degree;
    using Accum = typename Context::Accum;
    using Letter = typename Context::Letter;
    using Bitmask = typename Context::Bitmask;
    ignore_unused(ctx);

    Letter letters[Strategy::Architecture::max_depth];
    const auto relative_idx = elt_idx - basis.start_of_degree(elt_degree);
    basis.unpack_index_to_letters(letters, elt_degree, relative_idx);

    const auto right_min_deg = std::max<Degree>(
        0, std::max<Degree>(elt_degree - lhs.max_degree(), rhs.min_degree()));
    const auto right_max_deg = std::min<Degree>(
        elt_degree,
        std::min<Degree>(elt_degree - lhs.min_degree(), rhs.max_degree()));

    Accum acc{0};
    for (Bitmask mask{0}; mask < (Bitmask{1} << elt_degree); ++mask) {
        Index left_idx;
        Index right_idx;
        Degree left_deg;
        Degree right_deg;
        basis.pack_masked_index(letters,
                                elt_degree,
                                mask,
                                left_deg,
                                left_idx,
                                right_deg,
                                right_idx);
        left_idx += basis.start_of_degree(left_deg);
        right_idx += basis.start_of_degree(right_deg);

        if (lhs.has_degree(left_deg) && rhs.has_degree(right_deg)) {
            acc += Accum{lhs[left_idx]} * Accum{rhs[right_idx]};
        }
    }

    return acc;
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