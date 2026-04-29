#ifndef RPP_GPU_OPS_BLOCK_VECTOR_INPLACE_ADD_HPP
#define RPP_GPU_OPS_BLOCK_VECTOR_INPLACE_ADD_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class VectorInplaceAdd<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename VectorLhs, typename VectorRhs>
    RPP_DEVICE void operator()(Context const& ctx, VectorLhs& lhs, VectorRhs const& rhs, Accum alpha = Accum{1}) const noexcept {
        using Scalar = typename VectorLhs::value_type;
        auto const& basis = lhs.basis();
        const auto min_degree = std::max(lhs.min_degree(), rhs.min_degree());
        const auto max_degree = std::min(lhs.max_degree(), rhs.max_degree());
        if (max_degree < min_degree) {
            return;
        }

        const auto begin = basis.start_of_degree(min_degree);
        const auto size = basis.end_of_degree(max_degree) - begin;
        auto lhs_data = lhs.data() + begin;
        auto rhs_data = rhs.data() + begin;

        for (Index i = ctx.thread_rank(); i < size; i += ctx.num_threads()) {
            Accum lhs_val { lhs_data[i] };
            Accum rhs_val { rhs_data[i] };
            Accum result = lhs_val + alpha * rhs_val;
            lhs_data[i] = static_cast<Scalar>(result);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void vector_inplace_add_kernel(
    const BatchLhs batch_lhs,
    const BatchRhs batch_rhs,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::VectorInplaceAdd<Strategy> op;
    auto lhs = batch_lhs.view(my_index, basis);
    op(ctx, lhs, batch_rhs.view(my_index, basis), alpha);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_VECTOR_INPLACE_ADD_HPP
