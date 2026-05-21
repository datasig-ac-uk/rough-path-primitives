#ifndef RPP_OPERATIONS_BASIC_ST_INPLACE_FMA_HPP
#define RPP_OPERATIONS_BASIC_ST_INPLACE_FMA_HPP

#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {

template <typename Strategy, typename = void>
class STInplaceFma : public BaseOperation<Strategy> {
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    static constexpr bool is_implemented = false;

    template <typename TensorA, typename TensorB, typename TensorC>
    RPP_HOST_DEVICE void operator()(Context const& ctx,
                                    TensorA& a,
                                    TensorB const& b,
                                    TensorC const& c,
                                    Accum alpha = Accum{1},
                                    Accum beta = Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy,
                               Context,
                               TensorA,
                               TensorB,
                               TensorC,
                               Accum>,
            "rpp::ops::STInplaceFma has no implementation for this Strategy. "
            "Use an operation specialization for the selected strategy and "
            "include its header.");
    }
};

template <typename Strategy,
          typename BatchA,
          typename BatchB,
          typename BatchC,
          typename Basis>
auto st_inplace_fma(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchA const& a,
    BatchB const& b,
    BatchC const& c,
    Basis const& basis,
    typename Strategy::Index num_batches,
    typename Strategy::Accum alpha = typename Strategy::Accum{1},
    typename Strategy::Accum beta = typename Strategy::Accum{1}) noexcept {
    using Op = STInplaceFma<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"STInplaceFma\" that implements "
        "\"st_inplace_fma\" "
        "is not implemented. This either means that the Strategy object is "
        "invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include "
        "<rpp/cpu/single_thread/operations/basic/st_inplace_fma.hpp>");

    return strategy.template launch<Op>(std::move(config),
                                        std::make_tuple(a, b, c),
                                        make_basis_pack(basis),
                                        num_batches,
                                        alpha,
                                        beta);
}

} // namespace rpp::ops


#endif // RPP_OPERATIONS_BASIC_ST_INPLACE_FMA_HPP
