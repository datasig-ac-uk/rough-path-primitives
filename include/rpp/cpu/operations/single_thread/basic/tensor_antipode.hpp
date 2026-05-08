#ifndef RPP_CPU_OPS_SINGLE_THREAD_TENSOR_ANTIPODE_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_TENSOR_ANTIPODE_HPP

#include <rpp/cpu/operations/single_thread/basic/tensor_generalised_antipode.hpp>

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, typename Architecture>
void tensor_antipode_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::TensorAntipode<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index idx) {
            auto out = batch_out.view(idx, basis);
            auto arg = batch_arg.view(idx, basis);
            op(ctx, out, arg);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_TENSOR_ANTIPODE_HPP
