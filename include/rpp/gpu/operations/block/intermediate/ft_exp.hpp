#ifndef RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_EXP_HPP
#define RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_EXP_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/intermediate/ft_exp.hpp>

#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/operations/block/basic/ft_inplace_mul.hpp>
#include <rpp/gpu/operations/block/basic/tensor_add_identity.hpp>
#include <rpp/gpu/operations/block/basic/tensor_set_identity.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class FTExp<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;

    using InplaceMul = FTInplaceMul<Strategy>;
    using SetIdentity = TensorSetIdentity<Strategy>;
    using AddIdentity = TensorAddIdentity<Strategy>;

    InplaceMul inplace_mul;
    SetIdentity set_identity;
    AddIdentity add_identity;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return std::max(InplaceMul::scratch_space_size(strategy, basis),
                        std::max(SetIdentity::scratch_space_size(strategy, basis),
                                 AddIdentity::scratch_space_size(strategy, basis)));
    }

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one{1};
        set_identity(ctx, out);

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;
            ctx.sync();
            inplace_mul(ctx, out, arg.truncate(1, max_degree), divisor);
            add_identity(ctx, out);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void ft_exp_kernel(
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

    ops::FTExp<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto arg = batch_arg.view(my_index, basis);
    op(ctx, out, arg);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_INTERMEDIATE_FT_EXP_HPP
