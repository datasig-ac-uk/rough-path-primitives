#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_FT_ADJ_RMUL_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_FT_ADJ_RMUL_HPP

#include <algorithm>

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/operations/block/detail/ft_adjoint_multiply.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class FTAdjRMul<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return sizeof(typename Strategy::BlockReduceArray);
    }

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const& basis = out.basis();

        if (op.min_degree() == 0) {
            out[0] = static_cast<Scalar>(gpu::block::adjoint_low_degree_reduce<Accum>(
                ctx,
                op,
                arg,
                std::max(op.min_degree(), arg.min_degree()),
                std::min(op.max_degree(), arg.max_degree()),
                basis,
                [](Index i) { return i; }
            ));
        }

        const auto out_deg_min = std::max(Degree{1}, out.min_degree());
        const auto out_deg_max = out.max_degree();
        const auto begin = basis.start_of_degree(out_deg_min);
        const auto end = basis.end_of_degree(out_deg_max);
        for (Index elt_idx = begin + static_cast<Index>(ctx.thread_rank()); elt_idx < end; elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            const auto op_min_deg = std::max(op.min_degree(), static_cast<Degree>(arg.min_degree() - degree));
            const auto op_max_deg = std::min(op.max_degree(), static_cast<Degree>(arg.max_degree() - degree));

            Accum elt{0};
            for (Degree op_deg = op_min_deg; op_deg <= op_max_deg; ++op_deg) {
                const auto arg_stride = basis.size_of_degree(op_deg);
                for (Index op_idx = basis.start_of_degree(op_deg); op_idx < basis.end_of_degree(op_deg); ++op_idx) {
                    elt += Accum{arg[elt_idx * arg_stride + op_idx]} * Accum{op[op_idx]};
                }
            }

            out[elt_idx] = static_cast<Scalar>(elt);
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchOp, typename BatchArg, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void ft_adj_rmul_kernel(
    const BatchOut batch_out,
    const BatchOp batch_op,
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

    ops::FTAdjRMul<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto op_tensor = batch_op.view(my_index, basis);
    auto arg = batch_arg.view(my_index, basis);
    op(ctx, out, op_tensor, arg);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_FT_ADJ_RMUL_HPP
