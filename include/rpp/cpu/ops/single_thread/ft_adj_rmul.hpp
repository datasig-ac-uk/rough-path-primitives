#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_ADJ_RMUL_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_ADJ_RMUL_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/cpu/strategies.hpp>
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

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_ADJ_RMUL_HPP
