//
// Created by sam on 30/09/2026.
//

#ifndef RPP_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
#define RPP_OPERATIONS_BASIC_LEFT_HS_MUL_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>


#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {


template <typename Strategy, typename = void>
class LeftHSMul : public BaseOperation<Strategy> {

    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    static constexpr bool is_implemented = false;

    template <typename TensorOut,
              typename TensorIntegrand,
              typename TensorIntegrator>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorIntegrand const& integrand,
                    TensorIntegrator const& integrator,
                    Accum beta = Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy,
                               Context,
                               TensorOut,
                               TensorIntegrand,
                               TensorIntegrator,
                               Accum>,
            "rpp::ops::LeftHSMul has no implementation for this Strategy/Mul "
            "type. "
            "Use an operation specialization for the selected strategy and "
            "include its header.");
    }
};

template <typename Strategy,
          typename BatchOut,
          typename BatchIntegrand,
          typename BatchIntegrator,
          typename Basis>
auto left_hs_mul(Strategy const& strategy,
                 typename Strategy::LaunchConfig config,
                 BatchOut const& out,
                 BatchIntegrand const& integrand,
                 BatchIntegrator const& integrator,
                 Basis const& basis,
                 typename Strategy::Index num_batches,
                 typename Strategy::Accum beta = typename Strategy::Accum{
                     1}) noexcept {
    using Op = LeftHSMul<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"LeftHSMul\" that implements \"left_hs_mul\" "
        "is not implemented. This either means that the Strategy object is "
        "invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include "
        "<rpp/cpu/single_thread/operations/basic/left_hs_mul.hpp>");

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, integrand, integrator),
        make_basis_pack(basis),
        num_batches,
        beta);
}


} // namespace rpp::ops


#endif // RPP_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
