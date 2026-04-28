#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_FMEXP_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_FMEXP_HPP

#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

#include <rpp/cpu/ops/single_thread/vector_inplace_add.hpp>
#include <rpp/cpu/ops/single_thread/vector_assign.hpp>
#include <rpp/cpu/ops/single_thread/ft_inplace_fma.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class FTFMExp<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

    using Degree = typename Strategy::Degree;
    using Accum = typename Strategy::Accum;

    VectorAssign<Strategy> assign;
    VectorInplaceAdd<Strategy> inplace_add;
    FTInplaceMul<Strategy> inplace_mul;


public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorMultiplier, typename TensorExponent>
    void operator()(Context const& ctx, TensorOut& out, TensorMultiplier const& multiplier, TensorExponent const& exponent) const noexcept {
        auto const& basis = out.basis();
        constexpr Accum one { 1 };

        assign(ctx, out, multiplier);

        for (Degree d=basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;

            auto trunc_out = out.truncate(0, max_degree);
            const auto trunc_exponent = exponent.truncate(1, max_degree);
            inplace_mul(ctx, trunc_out, trunc_exponent, divisor);

            inplace_add(ctx, out, multiplier);
        }


    }
};

} // namespace rpp::ops

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_FMEXP_HPP
