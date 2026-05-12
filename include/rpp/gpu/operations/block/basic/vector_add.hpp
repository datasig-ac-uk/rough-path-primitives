#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_ADD_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_ADD_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/vector_add.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class VectorAdd<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > : public BaseOperation<
            gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    static constexpr bool is_implemented = true;

    template<typename VectorOut, typename VectorLhs, typename VectorRhs>
    RPP_DEVICE void operator()(Context const &ctx, VectorOut &out, VectorLhs const &lhs, VectorRhs const &rhs,
                               Accum alpha = Accum{1}, Accum beta = Accum{1}) const noexcept {
        using Scalar = typename VectorOut::value_type;
        auto const &basis = out.basis();
        const auto min_degree = std::max({out.min_degree(), lhs.min_degree(), rhs.min_degree()});
        const auto max_degree = std::min({out.max_degree(), lhs.max_degree(), rhs.max_degree()});
        if (max_degree < min_degree) {
            return;
        }

        const auto begin = basis.start_of_degree(min_degree);
        const auto size = basis.end_of_degree(max_degree) - begin;
        auto out_data = out.data() + begin;
        auto lhs_data = lhs.data() + begin;
        auto rhs_data = rhs.data() + begin;

        for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
            Accum lhs_val{lhs_data[i]};
            Accum rhs_val{rhs_data[i]};
            Accum result = alpha * lhs_val + beta * rhs_val;
            out_data[i] = static_cast<Scalar>(result);
        }
    }
};
} // namespace rpp::ops

namespace rpp::gpu::block {
template<typename BatchOut, typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, unsigned BlockSize,
    unsigned MaxBlockSize
    , typename Architecture>
RPP_KERNEL void vector_add_kernel(
    const BatchOut batch_out,
    const BatchLhs batch_lhs,
    const BatchRhs batch_rhs,
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

    ops::VectorAdd<Strategy> op;
    auto out = batch_out.view(my_index, basis);
    auto lhs = batch_lhs.view(my_index, basis);
    auto rhs = batch_rhs.view(my_index, basis);
    op(ctx, out, lhs, rhs, alpha, beta);
}
} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_VECTOR_ADD_HPP
