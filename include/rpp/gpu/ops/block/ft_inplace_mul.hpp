#ifndef RPP_GPU_OPS_BLOCK_FT_INPLACE_MUL_HPP
#define RPP_GPU_OPS_BLOCK_FT_INPLACE_MUL_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/ops/block/detail/ft_multiply.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class FTInplaceMul<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorLhs, typename TensorRhs>
    RPP_DEVICE void operator()(Context const& ctx, TensorLhs& lhs, TensorRhs const& rhs, Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorLhs::value_type;
        auto const& basis = lhs.basis();
        const auto low_range_degree = std::min(lhs.max_degree(), ctx.low_range_degree(basis));

        for (Degree out_deg = lhs.max_degree(); out_deg > low_range_degree; --out_deg) {
            for (auto elt_idx = basis.start_of_degree(out_deg) + ctx.thread_rank(); elt_idx < basis.end_of_degree(out_deg); elt_idx += ctx.num_threads()) {
                const auto acc = gpu::block::ft_multiply_loop_with_degree(ctx, lhs, rhs, elt_idx, out_deg, basis);
                lhs[elt_idx] = static_cast<Scalar>(beta * acc);
            }
            ctx.sync();
        }

        auto elt_idx = static_cast<Index>(ctx.thread_rank());
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);
        Accum acc{0};
        if (active) {
            acc = gpu::block::ft_multiply_loop_with_degree(ctx, lhs, rhs, elt_idx, basis.degree_linear(elt_idx), basis);
        }

        ctx.sync();
        if (active) {
            lhs[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void ft_inplace_mul_kernel(
    const BatchLhs batch_lhs,
    const BatchRhs batch_rhs,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::FTInplaceMul<Strategy> op;

    op(ctx, batch_lhs.view(my_index, basis), batch_rhs.view(my_index, basis), beta);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_FT_INPLACE_MUL_HPP
