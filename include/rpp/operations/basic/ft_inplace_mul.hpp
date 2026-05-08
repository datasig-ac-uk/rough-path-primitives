#ifndef RPP_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP
#define RPP_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class FTInplaceMul {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
public:
    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename TensorLhs, typename TensorRhs>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorLhs& lhs, TensorRhs const& rhs, Accum beta=Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorLhs, TensorRhs, Accum>,
            "rpp::ops::FTInplaceMul has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};


} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_INPLACE_MUL_HPP
