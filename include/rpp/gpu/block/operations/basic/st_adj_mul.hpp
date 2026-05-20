#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_ADJ_MUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_ADJ_MUL_HPP

#include <cuda/atomic>

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_adj_mul.hpp>

#include <rpp/gpu/block/strategy.hpp>
#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class STAdjMul<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize,
            Architecture> > : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize,
            Architecture> > {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

    using SetConstant = VectorSetConstant<Strategy>;

    SetConstant set_constant;

    using AtomicRef = cuda::atomic_ref<Accum>;
public:
    static constexpr bool is_implemented = true;

    template<typename TensorOut, typename TensorOp, typename TensorArg>
    RPP_DEVICE void operator()(Context const &ctx, TensorOut &out, TensorOp const &op,
                               TensorArg const &arg) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const &basis = out.basis();

        set_constant(ctx, out, Accum{0});
        ctx.sync();

        const auto arg_begin = basis.start_of_degree(arg.min_degree());
        const auto arg_end = basis.end_of_degree(arg.max_degree());
        for (Index elt_idx = arg_begin + ctx.thread_rank(); elt_idx < arg_end; elt_idx += ctx.num_threads()) {
            const auto elt_degree = basis.degree(elt_idx);
            Letter letters[Strategy::Architecture::max_depth];
            basis.unpack_index_to_letters(letters, elt_degree, elt_idx - basis.start_of_degree(elt_degree));
            const Accum arg_val{arg[elt_idx]};

            for (Bitmask mask{0}; mask < (Bitmask{1} << elt_degree); ++mask) {
                Index op_idx;
                Index out_idx;
                Degree op_deg;
                Degree out_deg;
                basis.pack_masked_index(
                    letters,
                    elt_degree,
                    mask,
                    op_deg,
                    op_idx,
                    out_deg,
                    out_idx
                );
                op_idx += basis.start_of_degree(op_deg);
                out_idx += basis.start_of_degree(out_deg);

                if (out.has_degree(out_deg) && op.has_degree(op_deg)) {
                    AtomicRef out_ref{out[out_idx]};
                    out_ref += static_cast<Scalar>(arg_val * Accum{op[op_idx]});
                }
            }
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_ADJ_MUL_HPP