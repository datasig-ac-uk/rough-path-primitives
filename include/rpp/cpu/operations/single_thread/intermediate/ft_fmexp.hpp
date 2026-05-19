#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_INTERMEDIATE_FT_FMEXP_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_INTERMEDIATE_FT_FMEXP_HPP

#include <cstddef>

#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/intermediate/ft_fmexp.hpp>

#include <rpp/cpu/operations/single_thread/strategy.hpp>
#include <rpp/cpu/operations/single_thread/linalg/vector_inplace_add.hpp>
#include <rpp/cpu/operations/single_thread/linalg/vector_assign.hpp>
#include <rpp/cpu/operations/single_thread/basic/ft_inplace_mul.hpp>

namespace rpp {
template<typename Accum_, typename Architecture>
class ops::FTFMExp<cpu::strategies::SingleThreadStrategy<Accum_,
            Architecture> > : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture> > {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

    using Degree = typename Strategy::Degree;
    using Accum = typename Strategy::Accum;

    VectorAssign<Strategy> assign;
    VectorInplaceAdd<Strategy> inplace_add;
    FTInplaceMul<Strategy> inplace_mul;

public:
    static constexpr bool is_implemented = true;

    template<typename TensorOut, typename TensorMultiplier, typename TensorExponent>
    void operator()(Context const &ctx, TensorOut &out, TensorMultiplier const &multiplier,
                    TensorExponent const &exponent) const noexcept {
        auto const &basis = out.basis();
        const Accum one{1};

        assign(ctx, out, multiplier);

        for (Degree d = basis.depth; d > 0; --d) {
            const Accum divisor = one / d;
            inplace_mul(ctx, out, exponent.truncate(1, basis.depth), divisor);
            inplace_add(ctx, out, multiplier);
        }
    }
};
} // namespace rpp

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_INTERMEDIATE_FT_FMEXP_HPP
