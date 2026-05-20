#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP

#include <rpp/config.h>
#include <rpp/views/batch.hpp>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_inplace_mul.hpp>

#include <rpp/gpu/block/strategy.hpp>
#include <rpp/gpu/block/operations/basic/detail/ft_multiply.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class FTInplaceMul<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > : public
        BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

    static constexpr bool is_implemented = true;

    template<typename TensorLhs, typename TensorRhs>
    RPP_DEVICE void operator()(Context const &ctx, TensorLhs &lhs, TensorRhs const &rhs,
                               Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorLhs::value_type;
        auto const &basis = lhs.basis();
        const auto low_range_degree = std::min(lhs.max_degree(), ctx.low_range_degree(basis));

        for (Degree out_deg = lhs.max_degree(); out_deg > low_range_degree; --out_deg) {
            for (auto elt_idx = basis.start_of_degree(out_deg) + ctx.thread_rank();
                 elt_idx < basis.end_of_degree(out_deg); elt_idx += ctx.num_threads()) {
                const auto acc = gpu::block::ft_multiply_loop_with_degree(ctx, lhs, rhs, elt_idx, out_deg, basis);
                lhs[elt_idx] = static_cast<Scalar>(beta * acc);
            }
            ctx.sync();
        }

        auto elt_idx = static_cast<Index>(ctx.thread_rank());
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);
        Accum acc{0};
        if (active) {
            acc = gpu::block::ft_multiply_loop_with_degree(ctx, lhs, rhs, elt_idx, basis.degree_linear(elt_idx), basis);
        }

        ctx.sync();
        if (active) {
            lhs[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP