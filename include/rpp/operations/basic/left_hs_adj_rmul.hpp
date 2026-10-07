#ifndef RPP_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
#define RPP_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP

#include <tuple>
#include <utility>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename = void>
class LeftHSAdjRMul : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    static constexpr bool is_implemented = false;

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    RPP_HOST_DEVICE void operator()(Context const&,
                                    TensorOut&,
                                    TensorOp const&,
                                    TensorArg const&,
                                    Accum = Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, TensorOut, TensorOp, TensorArg>,
            "Include the strategy-specific left_hs_adj_rmul.hpp "
            "implementation.");
    }
};

template <typename Strategy,
          typename BatchOut,
          typename BatchOp,
          typename BatchArg,
          typename Basis>
auto left_hs_adj_rmul(Strategy const& strategy,
                      typename Strategy::LaunchConfig config,
                      BatchOut const& out,
                      BatchOp const& op,
                      BatchArg const& arg,
                      Basis const& basis,
                      typename Strategy::Index num_batches,
                      typename Strategy::Accum beta = typename Strategy::Accum{
                          1}) noexcept {
    using Op = LeftHSAdjRMul<Strategy>;
    static_assert(
        Op::is_implemented,
        "Include the strategy-specific left_hs_adj_rmul.hpp implementation.");
    return strategy.template launch<Op>(std::move(config),
                                        std::make_tuple(out, op, arg),
                                        make_basis_pack(basis),
                                        num_batches,
                                        beta);
}

} // namespace rpp::ops

#endif // RPP_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
