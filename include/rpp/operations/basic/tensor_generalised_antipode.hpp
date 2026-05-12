#ifndef RPP_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {
enum class TensorAntipodeSigningPolicy {
    NoSigning,
    SignByDegree
};

template<typename Strategy, TensorAntipodeSigningPolicy Policy, typename=void>
class TensorGeneralisedAntipode : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;

public:
    static constexpr bool is_implemented = false;

    template<typename TensorOut, typename TensorArg>
    void operator()(Context const &ctx, TensorOut &out, TensorArg const &arg) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorOut, TensorArg>,
            "rpp::ops::TensorGeneralisedAntipode has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template<TensorAntipodeSigningPolicy Policy, typename Strategy, typename BatchOut, typename BatchArg, typename Basis>
auto tensor_generalised_antipode(
    Strategy const &strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const &out,
    BatchArg const &arg,
    Basis const &basis,
    typename Strategy::Index num_batches
) noexcept {
    using Op = TensorGeneralisedAntipode<Strategy, Policy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"TensorGeneralisedAntipode\" that implements "
        "\"tensor_generalised_antipode\" is not implemented. This either means "
        "that the Strategy object is invalid, or that the necessary specialisation "
        "headers have not been included. For example, you may need to add the "
        "following include directive to bring in the single-threaded CPU "
        "implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/tensor_generalised_antipode.hpp>"
    );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, arg),
        make_basis_pack(basis),
        num_batches
    );
}


template<typename Strategy>
using TensorAntipode = TensorGeneralisedAntipode<Strategy, TensorAntipodeSigningPolicy::SignByDegree>;

template<typename Strategy>
using TensorReflect = TensorGeneralisedAntipode<Strategy, TensorAntipodeSigningPolicy::NoSigning>;

} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
