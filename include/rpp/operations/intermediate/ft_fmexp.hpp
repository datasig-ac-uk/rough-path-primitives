#ifndef RPP_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP
#define RPP_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

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
    static constexpr bool is_implemented = InplaceFMA123::is_implemented && Assign::is_implemented;

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

template <typename Strategy, typename BatchOut, typename BatchMultiplier, typename BatchExponent, typename Basis>
auto ft_fmexp(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchMultiplier const& multiplier,
    BatchExponent const& exponent,
    Basis const& basis,
    typename Strategy::Index batch_size
    ) noexcept {
    using Op = FTFMExp<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"FTFMExp\" that implements \"ft_fmexp\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/intermediate/ft_fmexp.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, multiplier, exponent),
        basis,
        batch_size
        );
}

} // namespace rpp::ops



#endif //RPP_OPERATIONS_INTERMEDIATE_FT_FMEXP_HPP
