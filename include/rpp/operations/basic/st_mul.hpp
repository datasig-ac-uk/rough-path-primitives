#ifndef RPP_OPERATIONS_BASIC_ST_MUL_HPP
#define RPP_OPERATIONS_BASIC_ST_MUL_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_inplace_fma.hpp>

namespace rpp::ops {

template <typename Strategy, typename = void>
class STMul : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using InplaceFMA = STInplaceFma<Strategy>;
    InplaceFMA fma;

public:
    static constexpr bool is_implemented = InplaceFMA::is_implemented;

    template <typename Basis>
    static constexpr size_t scratch_space_size(Strategy const& strategy,
                                               Basis const& basis) noexcept {
        return InplaceFMA::scratch_space_size(strategy, basis);
    }

    template <typename TensorOut, typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorLhs const& lhs,
                    TensorRhs const& rhs,
                    Accum beta = Accum{1}) const noexcept {
        return fma(ctx, out, lhs, rhs, Accum{0}, beta);
    }
};

template <typename Strategy,
          typename BatchOut,
          typename BatchLhs,
          typename BatchRhs,
          typename Basis>
auto st_mul(Strategy const& strategy,
            typename Strategy::LaunchConfig config,
            BatchOut const& out,
            BatchLhs const& lhs,
            BatchRhs const& rhs,
            Basis const& basis,
            typename Strategy::Index num_batches,
            typename Strategy::Accum beta = typename Strategy::Accum{
                1}) noexcept {
    using Op = STMul<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"STMul\" that implements \"st_mul\" "
        "is not implemented. This either means that the Strategy object is "
        "invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/single_thread/operations/basic/st_mul.hpp>");

    return strategy.template launch<Op>(std::move(config),
                                        std::make_tuple(out, lhs, rhs),
                                        make_basis_pack(basis),
                                        num_batches,
                                        beta);
}

} // namespace rpp::ops

#endif // RPP_OPERATIONS_BASIC_ST_MUL_HPP
