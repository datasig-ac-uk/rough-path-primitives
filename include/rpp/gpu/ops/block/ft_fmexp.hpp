#ifndef RPP_GPU_OPS_BLOCK_FT_FMEXP_HPP
#define RPP_GPU_OPS_BLOCK_FT_FMEXP_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/operations.hpp>


#include <rpp/gpu/ops/block/vector_assign.hpp>
#include <rpp/gpu/ops/block/ft_inplace_fma.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class FTFMExp<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using InplaceFMA123 = FTInplaceFma123<Strategy>;
    using Assign = VectorAssign<Strategy>;

    InplaceFMA123 inplace_fma123;
    Assign assign;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return std::max(InplaceFMA123::scratch_space_size(strategy, basis), Assign::scratch_space_size(strategy, basis));
    }

    template <typename TensorOut, typename TensorMultiplier, typename TensorExponent>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorMultiplier const& multiplier, TensorExponent const& exponent) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one{1};
        assign(ctx, out, multiplier);

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;
            ctx.sync();
            inplace_fma123(ctx, out, exponent.truncate(1, max_degree), multiplier, one, divisor);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchMultiplier, typename BatchExponent, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void ft_fmexp_kernel(
    const BatchOut batch_out,
    const BatchMultiplier batch_multiplier,
    const BatchExponent batch_exponent,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::FTFMExp<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto multiplier = batch_multiplier.view(my_index, basis);
    auto exponent = batch_exponent.view(my_index, basis);
    op(ctx, out, multiplier, exponent);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_FT_FMEXP_HPP
