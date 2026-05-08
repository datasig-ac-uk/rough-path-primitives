#ifndef RPP_OPERATIONS_BASIC_FT_MUL_HPP
#define RPP_OPERATIONS_BASIC_FT_MUL_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/ft_inplace_fma.hpp>

namespace rpp::ops {

template <typename Strategy, typename=void>
class FTMul : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using FMA = FTInplaceFma<Strategy, FTInplaceFMAType::AEqualsBCPlusA>;
    FMA fma;

public:

    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        return FMA::scratch_space_size(strategy, basis);
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    RPP_HOST_DEVICE
    void operator()(Context const& ctx, TensorOut& out, TensorLhs const& lhs, TensorRhs& rhs, Accum beta=Accum{1}) const noexcept {
        fma(ctx, out, lhs, rhs, Accum{0}, beta);
    }
};



} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_MUL_HPP
