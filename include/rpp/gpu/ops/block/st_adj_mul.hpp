#ifndef RPP_GPU_OPS_BLOCK_ST_ADJ_MUL_HPP
#define RPP_GPU_OPS_BLOCK_ST_ADJ_MUL_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/gpu/ops/block/vector_set_zero.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class STAdjMul<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

    using SetZero = VectorSetZero<Strategy>;

    SetZero set_zero;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const& basis = out.basis();

        set_zero(ctx, out);
        ctx.sync();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < arg.size(); elt_idx += ctx.num_threads()) {
            const auto elt_degree = basis.degree(elt_idx);
            Letter letters[Strategy::Architecture::max_depth];
            basis.unpack_index_to_letters(letters, elt_degree, elt_idx);
            const Accum arg_val{arg[elt_idx]};

            for (Bitmask mask{0}; mask < Bitmask{1 << elt_degree}; ++mask) {
                Index op_idx;
                Index out_idx;
                Degree op_deg;
                Degree out_deg;
                basis.pack_masked_index(letters, elt_degree, mask, op_deg, op_idx, out_deg, out_idx);
                op_idx += basis.start_of_degree(op_deg);
                out_idx += basis.start_of_degree(out_deg);

                if (op.has_degree(op_deg)) {
                    atomicAdd(out.data() + out_idx, static_cast<Scalar>(arg_val * Accum{op[op_idx]}));
                }
            }
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchOp, typename BatchArg, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void st_adj_mul_kernel(
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

    ops::STAdjMul<Strategy> op;

    op(ctx, batch_out.view(my_index, basis), batch_op.view(my_index, basis), batch_arg.view(my_index, basis));
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPS_BLOCK_ST_ADJ_MUL_HPP
