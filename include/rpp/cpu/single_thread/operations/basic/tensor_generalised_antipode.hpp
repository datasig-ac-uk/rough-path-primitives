#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/tensor_generalised_antipode.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
namespace rpp::ops {


template <typename Accum_,
          TensorAntipodeSigningPolicy Policy,
          typename Architecture>
class TensorGeneralisedAntipode<
    cpu::strategies::SingleThreadStrategy<Accum_, Architecture>,
    Policy>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;

public:
    static constexpr bool is_implemented = true;

    template <typename TensorOut, typename TensorArg>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorArg const& arg) const noexcept {

        using Index = typename Context::Strategy::Index;
        const auto min_degree = std::max(out.min_degree(), arg.min_degree());
        const auto max_degree = std::min(out.max_degree(), arg.max_degree());
        auto const& basis = out.basis();

        if (min_degree == 0) {
            out[0] = arg[0];
        }

        for (auto degree = std::max(1, min_degree); degree <= max_degree;
             ++degree) {
            auto out_view = out.degree_view(degree);
            auto const arg_view = arg.degree_view(degree);

            for (Index i = 0; i < arg_view.size(); ++i) {
                auto const out_index = basis.reverse_index(i, degree);
                if constexpr (Policy ==
                              TensorAntipodeSigningPolicy::SignByDegree) {
                    using Value = std::decay_t<decltype(arg_view[i])>;
                    auto const sign =
                        degree % 2 == 0 ? Value{1} : Value{-1};
                    out_view[out_index] = arg_view[i] * sign;
                }
                else {
                    out_view[out_index] = arg_view[i];
                }
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_TENSOR_GENERALISED_ANTIPODE_HPP
