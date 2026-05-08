#ifndef RPP_OPERATIONS_BASIC_VECTOR_INPLACE_ADD_HPP
#define RPP_OPERATIONS_BASIC_VECTOR_INPLACE_ADD_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

namespace rpp::ops {

template<typename Strategy, typename=void>
class VectorInplaceAdd {
    using Accum = typename Strategy::Accum;

public:
    using Context = typename Strategy::Context;

    template<typename Basis>
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template<typename VectorLhs, typename VectorRhs>
    RPP_HOST_DEVICE
    void operator()(Context const &ctx, VectorLhs &lhs, VectorRhs const &rhs, Accum alpha = Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, VectorLhs, VectorRhs, Accum>,
            "rpp::ops::VectorInplaceAdd has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

}// namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_VECTOR_INPLACE_ADD_HPP
