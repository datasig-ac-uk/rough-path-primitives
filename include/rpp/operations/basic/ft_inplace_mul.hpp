#ifndef RPP_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP
#define RPP_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class FTInplaceMul : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    static constexpr bool is_implemented = false;

    template <typename TensorLhs, typename TensorRhs>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorLhs& lhs, TensorRhs const& rhs, Accum beta=Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorLhs, TensorRhs, Accum>,
            "rpp::ops::FTInplaceMul has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchLhs, typename BatchRhs, typename Basis>
auto ft_inplace_mul(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchLhs const& lhs,
    BatchRhs const& rhs,
    Basis const& basis,
    typename Strategy::Index batch_size,
    typename Strategy::Accum beta = typename Strategy::Accum{1}
    ) noexcept {
    using Op = FTInplaceMul<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"FTInplaceMul\" that implements \"ft_inplace_mul\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/ft_inplace_mul.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(lhs, rhs),
        basis,
        batch_size,
        beta
        );
}


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP
