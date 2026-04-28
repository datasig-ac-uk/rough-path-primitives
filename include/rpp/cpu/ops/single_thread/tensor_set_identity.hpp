#ifndef RPP_CPU_OPS_SINGLE_THREAD_TENSOR_SET_IDENTITY_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_TENSOR_SET_IDENTITY_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class TensorSetIdentity<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename Tensor>
    void operator()(Context const& ctx, Tensor& tensor, Accum scalar = Accum{1}) const noexcept {
        tensor[0] = scalar;
        std::fill(tensor.begin() + 1, tensor.end(), Accum{0});

    }
};

} // namespace rpp::ops

#endif // RPP_CPU_OPS_SINGLE_THREAD_TENSOR_SET_IDENTITY_HPP
