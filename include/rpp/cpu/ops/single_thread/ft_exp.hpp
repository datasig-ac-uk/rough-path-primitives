#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_EXP_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_EXP_HPP

#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/ops/single_thread/detail/batch_wrapper.hpp>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>


#include <rpp/cpu/ops/single_thread/ft_inplace_mul.hpp>
#include <rpp/cpu/ops/single_thread/tensor_add_identity.hpp>
#include <rpp/cpu/ops/single_thread/tensor_set_identity.hpp>


namespace rpp::ops {

// template <typename Accum_, typename Architecture>
// class FTExp<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
//     using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
//     using Context = typename Strategy::Context;
//
// public:
//     template <typename Basis>
//     static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
//         ignore_unused(strategy, basis);
//         return 0;
//     }
//
//     template <typename TensorOut, typename TensorArg>
//     void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
//         ignore_unused(ctx, out, arg);
//     }
// };

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, typename Architecture>
void ft_exp_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::FTExp<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto out = batch_out.view(tensor_idx, basis);
            auto arg = batch_arg.view(tensor_idx, basis);
            op(ctx, out, arg);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_EXP_HPP
