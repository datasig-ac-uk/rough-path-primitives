#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP

#include <algorithm>
#include <functional>

#include <rpp/config.h>


namespace rpp::gpu::block {

template <typename Context, typename TensorB, typename TensorC, typename Basis>
RPP_HOST_DEVICE constexpr auto
ft_multiply_loop_with_degree(const Context& ctx,
                             TensorB const& b,
                             TensorC const& c,
                             typename Context::Index elt_idx,
                             typename Context::Degree degree,
                             Basis const& basis) noexcept {
    using Index = typename Context::Index;
    using Degree = typename Context::Degree;
    using Accum = typename Context::Accum;
    ignore_unused(ctx);

    const auto degree_minus_b_max = degree >= b.max_degree()
        ? static_cast<Degree>(degree - b.max_degree())
        : Degree{0};
    const auto degree_minus_b_min = degree >= b.min_degree()
        ? static_cast<Degree>(degree - b.min_degree())
        : static_cast<Degree>(degree + 1);

    const auto rhs_min_deg = std::max(
        degree_minus_b_max, c.min_degree());
    const auto rhs_max_deg = std::min(
        degree_minus_b_min, c.max_degree());

    if (rhs_min_deg > rhs_max_deg || rhs_min_deg > degree) {
        return Accum{0};
    }

    Accum acc{0};
    if (rhs_min_deg == 0) {
        acc += Accum{b[elt_idx]} * Accum{c[0]};
    }

    const auto middle_min_deg = std::max<Degree>(rhs_min_deg, 1);
    const auto middle_max_deg = std::min<Degree>(degree - 1, rhs_max_deg);
    auto splitter = basis.size_of_degree(middle_min_deg);
    const Index idx = elt_idx - basis.degree_begin[degree];
    for (Degree rhs_deg = middle_min_deg; rhs_deg <= middle_max_deg;
         ++rhs_deg) {
        const auto lhs_deg = degree - rhs_deg;
        const auto lhs_idx = idx / splitter;
        const auto rhs_idx = idx % splitter;

        acc += Accum{b[basis.degree_begin[lhs_deg] + lhs_idx]} *
            Accum{c[basis.degree_begin[rhs_deg] + rhs_idx]};

        splitter *= basis.width;
    }

    if (degree > 0 && rhs_max_deg == degree) {
        acc += Accum{b[0]} * Accum{c[elt_idx]};
    }

    return acc;
}

template <typename Context, typename TensorB, typename TensorC, typename Basis>
RPP_HOST_DEVICE constexpr auto ft_multiply_loop(Context const& ctx,
                                                TensorB const& b,
                                                TensorC const& c,
                                                typename TensorB::Index elt_idx,
                                                Basis const& basis) noexcept {
    const auto degree = basis.degree(elt_idx);
    return ft_multiply_loop_with_degree(ctx, b, c, elt_idx, degree, basis);
}

} // namespace rpp::gpu::block


#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_DETAIL_FT_MULTIPLY_HPP
