#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_INPLACE_FMA_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_INPLACE_FMA_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/dense/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_inplace_fma.hpp>

#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/operations/block/basic/detail/st_multiply.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class STInplaceFma<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:

    template <typename TensorA, typename TensorB, typename TensorC>
    RPP_DEVICE void operator()(
        Context const& ctx,
        TensorA& a,
        TensorB const& b,
        TensorC const& c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) const noexcept {
        using Scalar = typename TensorA::value_type;
        auto const& basis = a.basis();
        for (Index elt_idx = ctx.thread_rank(); elt_idx < a.size(); elt_idx += ctx.num_threads()) {
            auto acc = gpu::block::st_multiply_loop(ctx, elt_idx, basis, b, c);
            acc *= beta;
            a[elt_idx] = static_cast<Scalar>(alpha * Accum{a[elt_idx]} + acc);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchA, typename BatchB, typename BatchC, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void st_inplace_fma_kernel(
    const BatchA batch_a,
    const BatchB batch_b,
    const BatchC batch_c,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1},
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::STInplaceFma<Strategy> op;

    auto a = batch_a.view(my_index, basis);
    auto b = batch_b.view(my_index, basis);
    auto c = batch_c.view(my_index, basis);
    op(ctx, a, b, c, alpha, beta);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_INPLACE_FMA_HPP
