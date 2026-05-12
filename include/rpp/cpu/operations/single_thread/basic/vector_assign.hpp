#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_ASSIGN_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_ASSIGN_HPP

#include <cstddef>
#include <algorithm>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/vector_assign.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/cpu/operations/single_thread/strategy.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class VectorAssign<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

public:
    static constexpr bool is_implemented = true;

    template <typename VectorOut, typename VectorArg>
    void operator()(Context const& ctx, VectorOut& out, VectorArg const& arg) const noexcept {
        auto const& basis = out.basis();
        const auto min_deg = std::max(out.min_degree(), arg.min_degree());
        const auto max_deg = std::min(out.max_degree(), arg.max_degree());

        const auto begin_index = basis.start_of_degree(min_deg);
        const auto end_index = basis.end_of_degree(max_deg);


        auto out_begin = out.data() + begin_index;
        auto arg_begin = arg.data() + begin_index;
        auto arg_end = arg.data() + end_index;


        std::copy(arg_begin, arg_end, out_begin);
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, typename Architecture>
void vector_assign_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::VectorAssign<Strategy>;

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

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_ASSIGN_HPP
