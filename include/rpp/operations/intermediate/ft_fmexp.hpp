#ifndef RPP_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP
#define RPP_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_inplace_fma.hpp>
#include <rpp/operations/basic/vector_assign.hpp>

namespace rpp::ops {


template <typename Strategy, typename=void>
class FTFMExp : public BaseOperation<Strategy> {

    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;


    using InplaceFMA123 = FTInplaceFma<Strategy, FTInplaceFMAType::AEqualsABPlusC>;
    using Assign = VectorAssign<Strategy>;

    InplaceFMA123 inplace_fma123;
    Assign assign;
public:
    using Context = typename Strategy::Context;

    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return std::max(
            InplaceFMA123 ::scratch_space_size(strategy, basis),
            Assign::scratch_space_size(strategy, basis));
    }

    template <typename TensorOut, typename TensorMultiplier, typename TensorExponent>
    void operator()(Context const& ctx, TensorOut& out, TensorMultiplier const& multiplier, TensorExponent const& exponent) const noexcept {
        auto const& basis = out.basis();
        const Accum one { 1 };

        assign(ctx, out, multiplier);

        for (Degree d =basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;

            ctx.sync();

            inplace_fma123(ctx, out, exponent.truncate(1, max_degree), multiplier, one, divisor);
        }
    }
};

} // namespace rpp::ops



#endif //RPP_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP
