#ifndef RPP_OPERATIONS_BASIC_VECTOR_ADD_HPP
#define RPP_OPERATIONS_BASIC_VECTOR_ADD_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template<typename Strategy, typename=void>
class VectorAdd : public BaseOperation<Strategy> {
    using Accum = typename Strategy::Accum;

public:
    using Context = typename Strategy::Context;
    static constexpr bool is_implemented = false;

    template<typename VectorOut, typename VectorLhs, typename VectorRhs> RPP_HOST_DEVICE
    void operator()(Context const &ctx, VectorOut &out, VectorLhs const &lhs, VectorRhs const &rhs,
                    Accum alpha = Accum{1}, Accum beta = Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, VectorLhs, VectorRhs, Accum>,
            "rpp::ops::VectorAdd has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchOut, typename BatchLhs, typename BatchRhs, typename Basis>
auto vector_add(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchLhs const& lhs,
    BatchRhs const& rhs,
    Basis const& basis,
    typename Strategy::Index num_batches,
    typename Strategy::Accum alpha = typename Strategy::Accum{1},
    typename Strategy::Accum beta = typename Strategy::Accum{1}
    ) noexcept {
    using Op = VectorAdd<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"VectorAdd\" that implements \"vector_add\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/single_thread/operations/linalg/vector_add.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, lhs, rhs),
        make_basis_pack(basis),
        num_batches,
        alpha,
        beta
        );
}

} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_VECTOR_ADD_HPP
