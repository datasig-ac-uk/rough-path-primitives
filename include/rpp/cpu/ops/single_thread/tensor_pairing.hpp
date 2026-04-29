#ifndef RPP_CPU_OPS_SINGLE_THREAD_TENSOR_PAIRING_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_TENSOR_PAIRING_HPP


#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

#include <rpp/cpu/strategies.hpp>


namespace rpp::ops {


template <typename Accum, typename Architecture>
class TensorPairing<cpu::strategies::SingleThreadStrategy<Accum, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum, Architecture>;
    using Context = typename Strategy::Context;

public:
    template <typename Basis>
    static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename Scalar, typename TensorFunc, typename TensorArg>
    void operator()(Context const& ctx, Scalar& out, TensorFunc const& functional, TensorArg const& arg) const noexcept {
        ignore_unused(ctx);
        auto const& basis = functional.basis();
        const auto min_degree = std::max(functional.min_degree(), arg.min_degree());
        const auto max_degree = std::min(functional.max_degree(), arg.max_degree());

        Accum tmp { 0 };
        for (auto i=basis.start_of_degree(min_degree); i < basis.end_of_degree(max_degree); ++i) {
            const Accum func_val { functional[i] };
            const Accum arg_val { arg[i] };
            tmp += func_val * arg_val;
        }

        out = tmp;
    }

};




}

#endif //RPP_CPU_OPS_SINGLE_THREAD_TENSOR_PAIRING_HPP
