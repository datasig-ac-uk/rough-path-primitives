#ifndef RPP_OPERATIONS_BASIC_TENSOR_PAIRING_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_PAIRING_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class TensorPairing {
public:
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

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

} // namespace rpp::ops
#endif //RPP_OPERATIONS_BASIC_TENSOR_PAIRING_HPP
