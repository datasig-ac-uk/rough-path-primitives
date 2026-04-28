#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_MUL_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_MUL_HPP

#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

#include <rpp/cpu/ops/single_thread/ft_fma.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class FTMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    FTFma<Strategy> fma;


public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs const& rhs, Accum beta = Accum{1}) const noexcept {
        fma(ctx, out, out, lhs, rhs, Accum{0}, beta);
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_MUL_HPP
