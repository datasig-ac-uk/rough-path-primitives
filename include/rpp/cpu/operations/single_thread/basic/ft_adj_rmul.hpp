#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_ADJ_RMUL_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_ADJ_RMUL_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/operations/basic/ft_adj_rmul.hpp>

#include <rpp/cpu/operations/single_thread/strategy.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>
#include <rpp/cpu/operations/single_thread/basic/tensor_antipode.hpp>
#include <rpp/cpu/operations/single_thread/basic/ft_adj_lmul.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class FTAdjRMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = Accum_;
    using Index = typename Strategy::Index;

    using Antipode = TensorAntipode<Strategy>;
    using AdjLMul = FTAdjLMul<Strategy>;

    Antipode antipode;
    AdjLMul adj_lmul;

    static Accum* batch_ptr(std::byte* base, Index batch_no, Index stride) noexcept{
        return reinterpret_cast<Accum*>(base + batch_no * stride);
    }

    static size_t batch_stride(Index basis_size) noexcept {
        return align_up(basis_size * sizeof(Accum), std::size_t{64});
    }

public:
    template <typename Basis>
    static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy);
        return 3 * batch_stride(basis.size());
    }

    template <typename Basis>
    static void init_scratch_space(Context const& ctx, Basis const& basis) noexcept {
        if constexpr (!std::is_trivially_constructible_v<Accum>) {
            const auto basis_size = basis.size();
            const auto stride = batch_stride(basis_size);
            auto* data = ctx.template scratch_space<std::byte*>();

            std::uninitialized_default_construct_n(batch_ptr(data, 0, stride), basis_size);
            std::uninitialized_default_construct_n(batch_ptr(data, 1, stride), basis_size);
            std::uninitialized_default_construct_n(batch_ptr(data, 2, stride), basis_size);
        }
    }

    template <typename Basis>
    static void destroy_scratch_space(Context const& ctx, Basis const& basis) noexcept {
        if constexpr (!std::is_trivially_destructible_v<Accum>) {
            const auto basis_size = basis.size();
            const auto stride = batch_stride(basis_size);
            auto* data = ctx.template scratch_space<std::byte*>();

            std::destroy_n(batch_ptr(data, 0, stride), basis_size);
            std::destroy_n(batch_ptr(data, 1, stride), basis_size);
            std::destroy_n(batch_ptr(data, 2, stride), basis_size);
        }
    }

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        auto* ptr = ctx.template scratch_space<std::byte*>();
        const auto stride = batch_stride(basis.size());
        dense::DenseTensorView<Accum*, typename TensorOut::Basis> out_workspace(batch_ptr(ptr, 0, stride), out.basis(), out.min_degree(), out.max_degree());
        dense::DenseTensorView<Accum*, typename TensorOp::Basis> op_workspace(batch_ptr(ptr, 1, stride), op.basis(), op.min_degree(), op.max_degree());
        dense::DenseTensorView<Accum*, typename TensorArg::Basis> arg_workspace(batch_ptr(ptr, 2, stride), arg.basis(), arg.min_degree(), arg.max_degree());

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

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_ADJ_RMUL_HPP
