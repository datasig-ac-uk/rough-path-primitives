#ifndef RPP_OPERATIONS_BASIC_TENSOR_PAIRING_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_PAIRING_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class TensorPairing : public BaseOperation<Strategy> {
public:
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    static constexpr bool is_implemented = false;


    template <typename Scalar, typename TensorFunc, typename TensorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, Scalar& result, TensorFunc const& functional, TensorArg const& arg) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, Scalar, TensorFunc, TensorArg>,
            "rpp::ops::TensorPairing has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchOut, typename FunctionalBatch, typename ArgBatch, typename Basis>
auto tensor_pairing(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& result,
    FunctionalBatch const& functional,
    ArgBatch const& arg,
    Basis const& basis,
    typename Strategy::Index batch_size
    ) noexcept {
    using Op = TensorPairing<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"TensorPairing\" that implements \"tensor_pairing\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/tensor_pairing.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(result, functional, arg),
        basis,
        batch_size
        );
}

} // namespace rpp::ops
#endif //RPP_OPERATIONS_BASIC_TENSOR_PAIRING_HPP
