#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_MUL_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_MUL_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>
#include <rpp/operations/basic/ft_mul.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>
#include <rpp/cpu/operations/single_thread/basic/ft_inplace_fma.hpp>

namespace rpp::ops {
//
// template <typename Accum_, typename Architecture>
// class FTMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
//     using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
//     using Context = typename Strategy::Context;
//     using Accum = typename Strategy::Accum;
//
//     FTFma<Strategy> fma;
//
//
// public:
//     template <typename Basis>
//     static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
//         ignore_unused(strategy, basis);
//         return 0;
//     }
//
//     template <typename TensorOut, typename TensorLhs, typename TensorRhs>
//     void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs const& rhs, Accum beta = Accum{1}) const noexcept {
//         fma(ctx, out, out, lhs, rhs, Accum{0}, beta);
//     }
// };

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, typename Architecture>
void ft_mul_kernel(
    const BatchOut batch_out,
    const BatchLhs batch_lhs,
    const BatchRhs batch_rhs,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::FTMul<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto out = batch_out.view(tensor_idx, basis);
            auto lhs = batch_lhs.view(tensor_idx, basis);
            auto rhs = batch_rhs.view(tensor_idx, basis);
            op(ctx, out, lhs, rhs, beta);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_MUL_HPP
