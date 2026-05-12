#ifndef RPP_OPERATIONS_BASIC_ST_ADJ_MUL_HPP
#define RPP_OPERATIONS_BASIC_ST_ADJ_MUL_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class STAdjMul : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
public:
    static constexpr bool is_implemented = false;

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, TensorOp const& op, TensorArg const& arg) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorOut, TensorOp, TensorArg>,
            "rpp::ops::STAdjMul has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchOut, typename BatchOp, typename BatchArg, typename Basis>
auto st_adj_mul(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchOp const& op,
    BatchArg const& arg,
    Basis const& basis,
    typename Strategy::Index num_batches
    ) noexcept {
    using Op = STAdjMul<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"STAdjMul\" that implements \"st_adj_mul\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/st_adj_mul.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, op, arg),
        make_basis_pack(basis),
        num_batches
        );
}

} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_ST_ADJ_MUL_HPP
