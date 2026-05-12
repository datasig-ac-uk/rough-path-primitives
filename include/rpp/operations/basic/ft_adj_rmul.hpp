#ifndef RPP_OPERATIONS_BASIC_FT_ADJ_RMUL_HPP
#define RPP_OPERATIONS_BASIC_FT_ADJ_RMUL_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {


template <typename Strategy, typename=void>
class FTAdjRMul : public BaseOperation<Strategy> {
public:
    static constexpr bool is_implemented = false;

    using Context = typename Strategy::Context;

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorOut, TensorOp, TensorArg>,
            "rpp::ops::FTAdjRMul has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchOut, typename BatchOp, typename BatchArg, typename Basis>
auto ft_adj_rmul(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchOp const& op,
    BatchArg const& arg,
    Basis const& basis,
    typename Strategy::Index batch_size
    ) noexcept {
    using Op = FTAdjRMul<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"FTAdjRMul\" that implements \"ft_adj_rmul\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/ft_adj_rmul.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, op, arg),
        basis,
        batch_size
        );
}


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_ADJ_RMUL_HPP
