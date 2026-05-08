#ifndef RPP_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

namespace rpp::ops {

enum class TensorAntipodeSigningPolicy {
    NoSigning,
    SignByDegree
};

template <typename Strategy, TensorAntipodeSigningPolicy Policy, typename=void>
class TensorGeneralisedAntipode {
    using Context = typename Strategy::Context;
public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename TensorOut, typename TensorArg>
    void operator()(Context const& ctx, TensorOut& out, TensorArg const& arg) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorOut, TensorArg>,
            "rpp::ops::TensorGeneralisedAntipode has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};


template <typename Strategy>
using TensorAntipode = TensorGeneralisedAntipode<Strategy, TensorAntipodeSigningPolicy::SignByDegree>;

template <typename Strategy>
using TensorReflect = TensorGeneralisedAntipode<Strategy, TensorAntipodeSigningPolicy::NoSigning>;

}// namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
