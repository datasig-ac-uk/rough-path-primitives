#ifndef RPP_GPU_OPS_BLOCK_FT_MUL_HPP
#define RPP_GPU_OPS_BLOCK_FT_MUL_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/gpu/ops/block/detail/ft_multiply.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class FTMul<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs const& rhs, Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const& basis = out.basis();
        for (Index elt_idx = ctx.thread_rank(); elt_idx < out.size(); elt_idx += ctx.num_threads()) {
            auto acc = gpu::block::ft_multiply_loop(ctx, lhs, rhs, elt_idx, basis);
            out[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void ft_mul_kernel(
    const BatchOut batch_out,
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

    ops::FTMul<Strategy> op;

    op(ctx, batch_out.view(my_index, basis), batch_lhs.view(my_index, basis), batch_rhs.view(my_index, basis), beta);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_FT_MUL_HPP
