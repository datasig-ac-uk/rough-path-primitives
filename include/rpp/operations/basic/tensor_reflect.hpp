#ifndef RPP_OPERATIONS_BASIC_TENSOR_REFLECT_HPP
#define RPP_OPERATIONS_BASIC_TENSOR_REFLECT_HPP

#include <tuple>
#include <utility>

#include <rpp/operations/basic/tensor_generalised_antipode.hpp>

namespace rpp::ops {

template <typename Strategy, typename BatchOut, typename BatchArg, typename Basis>
auto tensor_reflect(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchArg const& arg,
    Basis const& basis,
    typename Strategy::Index num_batches
    ) noexcept {
    using Op = TensorReflect<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"TensorReflect\" that implements \"tensor_reflect\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/tensor_reflect.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, arg),
        make_basis_pack(basis),
        num_batches
        );
}

} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_TENSOR_REFLECT_HPP
