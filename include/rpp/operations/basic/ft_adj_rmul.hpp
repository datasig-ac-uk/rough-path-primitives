#ifndef RPP_OPERATIONS_BASIC_FT_ADJ_RMUL_HPP
#define RPP_OPERATIONS_BASIC_FT_ADJ_RMUL_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {


template <typename Strategy, typename=void>
class FTAdjRMul : public BaseOperation<Strategy> {
public:
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


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_ADJ_RMUL_HPP
