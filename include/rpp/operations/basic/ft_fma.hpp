#ifndef RPP_OPERATIONS_BASIC_FT_FMA_HPP
#define RPP_OPERATIONS_BASIC_FT_FMA_HPP
#include <cstddef>
#include <tuple>
#include <utility>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>

namespace rpp::ops {
template<typename Strategy, typename=void>
class FTFma : public BaseOperation<Strategy>{
public:
    static constexpr bool is_implemented = false;

    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    template<typename TensorOut, typename TensorA, typename TensorB, typename TensorC>
    RPP_HOST_DEVICE
    void operator()(Context const &ctx, TensorOut &out, TensorA const &a, TensorB const &b, TensorC const &c,
                    Accum alpha = Accum{1}, Accum beta = Accum{1}) const noexcept {
        static_assert(
            static_assert_fail<Strategy, Context, TensorOut, TensorA, TensorB, TensorC, Accum>,
            "rpp::ops::FTFma has no implementation for this Strategy/FMA type. "
            "Use an operation specialization for the selected strategy and include its header."
        );
    }
};

template <typename Strategy, typename BatchOut, typename BatchA, typename BatchB, typename BatchC, typename Basis>
auto ft_fma(
    Strategy const& strategy,
    typename Strategy::LaunchConfig config,
    BatchOut const& out,
    BatchA const& a,
    BatchB const& b,
    BatchC const& c,
    Basis const& basis,
    typename Strategy::Index num_batches,
    typename Strategy::Accum alpha = typename Strategy::Accum{1},
    typename Strategy::Accum beta = typename Strategy::Accum{1}
    ) noexcept {
    using Op = FTFma<Strategy>;

    static_assert(
        Op::is_implemented,
        "The operation object \"FTFma\" that implements \"ft_fma\" "
        "is not implemented. This either means that the Strategy object is invalid, "
        "or that the necessary specialisation headers have not been included. "
        "For example, you may need to add the following include directive to "
        "bring in the single-threaded CPU implementation of this operation:\n\n"
        "    #include <rpp/cpu/operations/single_thread/basic/ft_fma.hpp>"
        );

    return strategy.template launch<Op>(
        std::move(config),
        std::make_tuple(out, a, b, c),
        make_basis_pack(basis),
        num_batches,
        alpha,
        beta
        );
}
} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_FMA_HPP
