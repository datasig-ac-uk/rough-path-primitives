#ifndef RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_LOG_HPP
#define RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_LOG_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/gpu/strategies.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/intermediate/ft_log.hpp>

#include <rpp/gpu/operations/block/basic/vector_set_constant.hpp>
#include <rpp/gpu/operations/block/basic/ft_inplace_mul.hpp>
#include <rpp/gpu/operations/block/basic/tensor_add_identity.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class FTLog<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using SetZero = VectorSetConstant<Strategy>;
    using InplaceMul = FTInplaceMul<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    SetZero set_zero;
    InplaceMul inplace_mul;
    AddIdentity add_identity;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return std::max(SetZero::scratch_space_size(strategy, basis),
                        std::max(InplaceMul::scratch_space_size(strategy, basis),
                                 AddIdentity::scratch_space_size(strategy, basis)));
    }

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one{1};
        set_zero(ctx, out, Accum{0});

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum val = (d % 2 == 0 ? -one : one) / d;
            add_identity(ctx, out, val);
            ctx.sync();
            auto trunc_out = out.truncate(0, max_degree);
            inplace_mul(ctx, trunc_out, arg.truncate(1, max_degree));
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void ft_log_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::FTLog<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto arg = batch_arg.view(my_index, basis);
    op(ctx, out, arg);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_LOG_HPP
