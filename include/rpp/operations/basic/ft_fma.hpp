#ifndef RPP_OPERATIONS_BASIC_FT_FMA_HPP
#define RPP_OPERATIONS_BASIC_FT_FMA_HPP
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

namespace rpp::ops {
template<typename Strategy, typename=void>
class FTFma {
    using Context = typename Strategy::Context;

    using Accum = typename Strategy::Accum;

public:
    template<typename Basis>
    static constexpr size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

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
} // namespace rpp::ops

#endif //RPP_OPERATIONS_BASIC_FT_FMA_HPP
