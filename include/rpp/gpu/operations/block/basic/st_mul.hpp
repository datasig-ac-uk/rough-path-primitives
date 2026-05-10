#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_MUL_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_MUL_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/dense/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_mul.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>
#include <rpp/gpu/operations/block/basic/detail/st_multiply.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class STMul<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > : public BaseOperation<
            gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    template<typename TensorOut, typename TensorLhs, typename TensorRhs>
    RPP_DEVICE void operator()(Context const &ctx, TensorOut &out, TensorLhs const &lhs, TensorRhs const &rhs,
                               Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const &basis = out.basis();
        for (Index elt_idx = ctx.thread_rank(); elt_idx < out.size(); elt_idx += ctx.num_threads()) {
            auto acc = gpu::block::st_multiply_loop(ctx, elt_idx, basis, lhs, rhs);
            out[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }
};
} // namespace rpp::ops

namespace rpp::gpu::block {
template<typename BatchOut, typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, unsigned BlockSize,
    unsigned MaxBlockSize
    , typename Architecture>
RPP_KERNEL void st_mul_kernel(
    const BatchOut batch_out,
    const BatchLhs batch_lhs,
    const BatchRhs batch_rhs,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::STMul<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto lhs = batch_lhs.view(my_index, basis);
    auto rhs = batch_rhs.view(my_index, basis);
    op(ctx, out, lhs, rhs, beta);
}
} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_MUL_HPP
