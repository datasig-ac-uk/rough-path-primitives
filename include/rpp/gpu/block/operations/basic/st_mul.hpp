#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_MUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_MUL_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_mul.hpp>

#include <rpp/gpu/block/operations/basic/detail/st_multiply.hpp>
#include <rpp/gpu/block/strategy.hpp>

namespace rpp::ops {
template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class STMul<gpu::strategies::
                BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    static constexpr bool is_implemented = true;

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorLhs const& lhs,
                               TensorRhs const& rhs,
                               Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const& basis = out.basis();
        for (Index elt_idx = ctx.thread_rank(); elt_idx < out.size();
             elt_idx += ctx.num_threads()) {
            auto acc =
                gpu::block::st_multiply_loop(ctx, elt_idx, basis, lhs, rhs);
            out[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_MUL_HPP