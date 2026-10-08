#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_LMUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_LMUL_HPP

// IWYU pragma: always_keep

#include <rpp/config.h>

#include <rpp/support/span.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
#include <rpp/operations/basic/left_hs_adj_lmul.hpp>
#include <rpp/operations/implementation/word_half_shuffle_product.hpp>

#include <rpp/operations/implementation/shuffle_adjoint_op_loop.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class LeftHSAdjLMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {

    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;
    using Letter = typename Strategy::Letter;
    using LetterSpan = Span<Letter>;

public:
    using Context = typename Strategy::Context;

private:
    template <typename TensorIntegrator, typename TensorIntegrand>
    static Accum letter_loop(Context const& ctx,
                             Index out_index,
                             Index out_degree,
                             LetterSpan letters,
                             TensorIntegrator const& integrator,
                             TensorIntegrand const& integrand,
                             Degree op_min_deg,
                             Degree op_max_deg) noexcept {

        const auto width = static_cast<Index>(integrator.basis().width);

        Accum acc{0};
        for (Index prefix = 0; prefix < width; ++prefix) {

            common::detail::PrefixLetterGetter<TensorIntegrand>
                integrand_getter{prefix};
            common::detail::PrefixLetterGetter<TensorIntegrator>
                integrator_getter{prefix};

            acc += common::shuffle_adjoint_op_loop(
                ctx,
                out_index,
                out_degree,
                letters,
                integrator,
                integrand,
                op_min_deg,
                op_max_deg,
                integrand_getter,
                integrator_getter,
                [](auto const& op_val, auto const& arg_val) -> Accum {
                    return Accum{arg_val} * Accum{op_val};
                });
        }

        return acc;
    }


public:
    static constexpr bool is_implemented = true;

    template <typename TensorOut,
              typename TensorIntegrator,
              typename TensorIntegrand>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorIntegrator const& integrator,
                    TensorIntegrand const& integrand) const noexcept {
        using Scalar = typename TensorOut::value_type;

        auto const& basis = out.basis();
        Letter letter_data[Strategy::Architecture::max_depth];
        LetterSpan letters{letter_data, Strategy::Architecture::max_depth};

        for (Degree out_degree = out.min_degree();
             out_degree <= out.max_degree();
             ++out_degree) {
            auto out_level = out.degree_view(out_degree);
            const auto op_min_deg =
                std::max({Degree{0},
                          integrator.min_degree() - 1,
                          integrand.min_degree() - 1 - out_degree});
            const auto op_max_deg =
                std::min(integrator.max_degree() - 1,
                         integrand.max_degree() - 1 - out_degree);

            for (Index i = 0; i < out_level.size(); ++i) {
                basis.unpack_index_to_letters(letters, out_degree, i);

                auto acc = letter_loop(ctx,
                                       i,
                                       out_degree,
                                       letters,
                                       integrator,
                                       integrand,
                                       op_min_deg,
                                       op_max_deg);

                out_level[i] = static_cast<Scalar>(acc);
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_LMUL_HPP
