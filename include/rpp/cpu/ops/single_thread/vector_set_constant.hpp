#ifndef RPP_CPU_OPS_SINGLE_THREAD_VECTOR_SET_CONSTANT_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_VECTOR_SET_CONSTANT_HPP

#include <algorithm>
#include <cstddef>


#include <rpp/cpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class VectorSetConstant<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename Vector, typename Value>
    void operator()(Context const& ctx, Vector& vec, Value const& value) const noexcept {
        std::fill(vec.begin(), vec.end(), value);
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_OPS_SINGLE_THREAD_VECTOR_SET_CONSTANT_HPP
