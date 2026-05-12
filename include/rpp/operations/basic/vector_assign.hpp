#ifndef RPP_OPERATIONS_BASIC_VECTOR_ASSIGN_HPP
#define RPP_OPERATIONS_BASIC_VECTOR_ASSIGN_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class VectorAssign : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
public:
    static constexpr bool is_implemented = false;

    template <typename VectorOut, typename VectorArg>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, VectorOut& out, VectorArg const& arg) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, VectorOut, VectorArg>,
            "rpp::ops::VectorAssign has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchOut, typename BatchArg, typename Basis>
auto vector_assign(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchArg const& arg,
    Basis const& basis,
    typename Strategy::Index num_batches
    ) noexcept {
    using Op = VectorAssign<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"VectorAssign\" that implements \"vector_assign\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/vector_assign.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, arg),
        make_basis_pack(basis),
        num_batches
        );
}

} // namespace rpp::ops


#endif //RPP_OPERATIONS_BASIC_VECTOR_ASSIGN_HPP
