#ifndef RPP_OPERATIONS_BASIC_VECTOR_SET_CONSTANT_HPP
#define RPP_OPERATIONS_BASIC_VECTOR_SET_CONSTANT_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class VectorSetConstant : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    static constexpr bool is_implemented = false;

    template <typename Vector>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, Vector& vec, Accum const& value=Accum{}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, Vector>,
            "rpp::ops::VectorSetConstant has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchVector, typename Basis>
auto vector_set_constant(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchVector const& vec,
    Basis const& basis,
    typename Strategy::Index num_batches,
    typename Strategy::Accum value = typename Strategy::Accum{}
    ) noexcept {
    using Op = VectorSetConstant<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"VectorSetConstant\" that implements "
        "\"vector_set_constant\" is not implemented. This either means that the "
        "Strategy object is invalid, or that the necessary specialisation headers "
        "have not been included. For example, you may need to add the following "
        "include directive to bring in the single-threaded CPU implementation of "
        "this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/vector_set_constant.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(vec),
        make_basis_pack(basis),
        num_batches,
        value
        );
}

} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_VECTOR_SET_CONSTANT_HPP
