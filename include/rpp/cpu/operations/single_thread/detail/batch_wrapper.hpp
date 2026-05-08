#ifndef RPP_CPU_OPS_SINGLE_THREAD_DETAIL_BATCH_WRAPPER_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_DETAIL_BATCH_WRAPPER_HPP

#include <cstddef>
#include <vector>

namespace rpp::cpu::single_thread::detail {

template <typename Op, typename Strategy, typename Basis, typename Fn>
void apply_batch(
    Basis const& basis,
    Strategy const& strategy,
    typename Strategy::Index n_tensors,
    Fn&& fn
) {
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    const auto scratch_size = Op::scratch_space_size(strategy, basis);
    std::vector<std::byte> scratch(scratch_size);
    auto const ctx = strategy.make_context(scratch.data());

    Op::init_scratch_space(ctx, basis);

    Op op;
    for (Index idx = 0; idx < n_tensors; ++idx) {
        fn(op, ctx, idx);
    }

    Op::destroy_scratch_space(ctx, basis);
}

} // namespace rpp::cpu::single_thread::detail

#endif // RPP_CPU_OPS_SINGLE_THREAD_DETAIL_BATCH_WRAPPER_HPP
