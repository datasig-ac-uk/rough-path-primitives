#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_SET_CONSTANT_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_SET_CONSTANT_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/linalg/vector_set_constant.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
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

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_SET_CONSTANT_HPP