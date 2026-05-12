#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_FMA_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_FMA_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_fma.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>
#include <rpp/gpu/operations/block/basic/detail/st_multiply.hpp>


namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class STFma<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > : public BaseOperation<
            gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    static constexpr bool is_implemented = true;

    template<typename TensorOut, typename TensorA, typename TensorB, typename TensorC>
    RPP_DEVICE void operator()(
        Context const &ctx,
        TensorOut &out,
        TensorA const &a,
        TensorB const &b,
        TensorC const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const &basis = out.basis();
        for (Index elt_idx = ctx.thread_rank(); elt_idx < out.size(); elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            auto acc = gpu::block::st_multiply_loop_with_degree(ctx, elt_idx, degree, basis, b, c);
            acc *= beta;
            Accum a_val{0};
            if (a.has_degree(degree)) {
                a_val = Accum{a[elt_idx]};
            }
            out[elt_idx] = static_cast<Scalar>(alpha * a_val + acc);
        }
    }
};
} // namespace rpp::ops

namespace rpp::gpu::block {
template<typename BatchOut, typename BatchA, typename BatchB, typename BatchC, typename Basis, typename Accum_,
    unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void st_fma_kernel(
    const BatchOut batch_out,
    const BatchA batch_a,
    const BatchB batch_b,
    const BatchC batch_c,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1},
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::STFma<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto a = batch_a.view(my_index, basis);
    auto b = batch_b.view(my_index, basis);
    auto c = batch_c.view(my_index, basis);
    op(
        ctx,
        out,
        a,
        b,
        c,
        alpha,
        beta
    );
}
} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_FMA_HPP
