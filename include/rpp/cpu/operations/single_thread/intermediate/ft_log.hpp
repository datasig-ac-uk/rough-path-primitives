#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_INTERMEDIATE_FT_LOG_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_INTERMEDIATE_FT_LOG_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/intermediate/ft_log.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>
#include <rpp/cpu/operations/single_thread/basic/vector_set_constant.hpp>
#include <rpp/cpu/operations/single_thread/basic/tensor_add_identity.hpp>
#include <rpp/cpu/operations/single_thread/basic/ft_inplace_mul.hpp>

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, typename Architecture>
void ft_log_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::FTLog<Strategy>;

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

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_INTERMEDIATE_FT_LOG_HPP
