#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_ADJ_RMUL_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_ADJ_RMUL_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/ops/single_thread/detail/batch_wrapper.hpp>
#include <rpp/dense/batch.hpp>
#include <rpp/dense/views.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

#include <rpp/cpu/ops/single_thread/tensor_antipode.hpp>
#include <rpp/cpu/ops/single_thread/ft_adj_lmul.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class FTAdjRMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = Accum_;

    using Antipode = TensorAntipode<Strategy>;
    using AdjLMul = FTAdjLMul<Strategy>;

    Antipode antipode;
    AdjLMul adj_lmul;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config);
        return 3 * align_up(basis.size() * sizeof(Accum), std::size_t{64});
    }

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        auto* ptr = ctx.template scratch_space<Accum>();
        const auto stride = align_up(basis.size() * sizeof(Accum), std::size_t{64}) / sizeof(Accum);
        dense::DenseTensorView<Accum*, typename TensorOut::Basis> out_workspace(ptr, out.basis(), out.min_degree(), out.max_degree());
        dense::DenseTensorView<Accum*, typename TensorOp::Basis> op_workspace(ptr + stride, op.basis(), op.min_degree(), op.max_degree());
        dense::DenseTensorView<Accum*, typename TensorArg::Basis> arg_workspace(ptr + 2 * stride, arg.basis(), arg.min_degree(), arg.max_degree());

        std::fill(out_workspace.begin(), out_workspace.end(), Accum{0});

        antipode(ctx, op_workspace, op);
        antipode(ctx, arg_workspace, arg);

        adj_lmul(ctx, out_workspace, op_workspace, arg_workspace);

        antipode(ctx, out, out_workspace);
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchOp, typename BatchArg, typename Basis, typename Accum_, typename Architecture>
void ft_adj_rmul_kernel(
    const BatchOut batch_out,
    const BatchOp batch_op,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::FTAdjRMul<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto out = batch_out.view(tensor_idx, basis);
            auto op_arg = batch_op.view(tensor_idx, basis);
            auto arg = batch_arg.view(tensor_idx, basis);
            op(ctx, out, op_arg, arg);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_ADJ_RMUL_HPP
