#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_DETAIL_FT_ADJOINT_MULTIPLY_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_DETAIL_FT_ADJOINT_MULTIPLY_HPP

#include <functional>

#include <rpp/config.h>

namespace rpp::gpu::block {

template <typename Accum, typename Context, typename TensorOp, typename TensorArg, typename Basis, typename ArgIndexFn>
RPP_DEVICE static Accum adjoint_low_degree_reduce(
    Context const& ctx,
    TensorOp const& op,
    TensorArg const& arg,
    typename TensorOp::Degree op_degree_min,
    typename TensorOp::Degree op_degree_max,
    Basis const& basis,
    ArgIndexFn&& arg_idx
) noexcept {
    using Degree = typename TensorOp::Degree;
    using Index = typename TensorOp::Index;

    Accum val{0};
    const auto begin = basis.start_of_degree(op_degree_min);
    const auto end = basis.end_of_degree(op_degree_max);
    for (Index i = begin + static_cast<Index>(ctx.thread_rank()); i < end; i += ctx.num_threads()) {
        val += Accum{op[i]} * Accum{arg[arg_idx(i)]};
    }
    return ctx.reduce(val, std::plus<Accum>{});
}

} // namespace rpp::gpu::block


#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_DETAIL_FT_ADJOINT_MULTIPLY_HPP
