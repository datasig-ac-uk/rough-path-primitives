#ifndef RPP_CPU_OPS_SINGLE_THREAD_VECTOR_SET_ZERO_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_VECTOR_SET_ZERO_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/ops/single_thread/detail/batch_wrapper.hpp>
#include <rpp/dense/batch.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class VectorSetZero<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename Vector>
    void operator()(Context const& ctx, Vector& vec) const noexcept {
        std::fill(vec.begin(), vec.end(), Accum{0});
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchVector, typename Basis, typename Accum_, typename Architecture>
void vector_set_zero_kernel(
    const BatchVector batch_vec,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::VectorSetZero<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto vec = batch_vec.view(tensor_idx, basis);
            op(ctx, vec);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_VECTOR_SET_ZERO_HPP
