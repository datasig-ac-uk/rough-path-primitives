#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_SET_CONSTANT_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_SET_CONSTANT_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/linalg/vector_set_constant.hpp>

#include <rpp/cpu/operations/single_thread/strategy.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class VectorSetConstant<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

public:
    static constexpr bool is_implemented = true;

    template <typename Vector, typename Value>
    void operator()(Context const& ctx, Vector& vec, Value const& value) const noexcept {
        std::fill(vec.begin(), vec.end(), value);
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchVector, typename Basis, typename Value, typename Accum_, typename Architecture>
void vector_set_constant_kernel(
    const BatchVector batch_vec,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors,
    const Value value
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::VectorSetConstant<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto vec = batch_vec.view(tensor_idx, basis);
            op(ctx, vec, value);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_VECTOR_SET_CONSTANT_HPP
