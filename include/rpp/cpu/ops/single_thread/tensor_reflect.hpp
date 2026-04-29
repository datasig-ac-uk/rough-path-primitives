#ifndef RPP_CPU_OPS_SINGLE_THREAD_TENSOR_REFLECT_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_TENSOR_REFLECT_HPP

#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/ops/single_thread/detail/batch_wrapper.hpp>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

#include <rpp/cpu/ops/single_thread/detail/antipode.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class TensorReflect<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        cpu::single_thread::generalised_antipode(ctx, out, arg, cpu::single_thread::NoSigningPolicy{});
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, typename Architecture>
void tensor_reflect_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::TensorReflect<Strategy>;

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

#endif // RPP_CPU_OPS_SINGLE_THREAD_TENSOR_REFLECT_HPP
