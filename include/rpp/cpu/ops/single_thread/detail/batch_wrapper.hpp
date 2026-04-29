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
    std::vector<Accum> scratch((scratch_size + sizeof(Accum) - 1) / sizeof(Accum));
    auto const ctx = strategy.make_context(reinterpret_cast<std::byte*>(scratch.data()));

    Op op;
    for (Index tensor_idx = 0; tensor_idx < n_tensors; ++tensor_idx) {
        fn(op, ctx, tensor_idx);
    }
}

} // namespace rpp::cpu::single_thread::detail

#endif // RPP_CPU_OPS_SINGLE_THREAD_DETAIL_BATCH_WRAPPER_HPP
