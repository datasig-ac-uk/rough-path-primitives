#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_LMUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_LMUL_HPP

// IWYU pragma: always_keep

#include <rpp/gpu/block/strategy.hpp>
#include <rpp/operations/basic/left_hs_adj_lmul.hpp>

#include <rpp/operations/implementation/shuffle_adjoint_op_loop.hpp>
#include <rpp/operations/implementation/word_half_shuffle_product.hpp>


namespace rpp::ops {

template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class LeftHSAdjLMul<
    gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {

public:
    using Strategy = gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;

    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;
    using Letter = typename Strategy::Letter;
    using LetterSpan = Span<Letter>;

public:
    using Context = typename Strategy::Context;

    static constexpr bool is_implemented = true;

private:
    template <typename TensorIntegrator, typename TensorIntegrand>
    RPP_DEVICE static Accum letter_loop(Context const& ctx,
                                        Index out_index,
                                        Degree out_degree,
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


    template <typename Basis,
              typename TensorIntegrator,
              typename TensorIntegrand>
    RPP_DEVICE static Accum evaluate(Context const& ctx,
                                     Degree out_degree,
                                     Index out_index,
                                     LetterSpan letters,
                                     Basis const& basis,
                                     TensorIntegrator const& integrator,
                                     TensorIntegrand const& integrand,
                                     Degree op_min_deg,
                                     Degree op_max_deg) noexcept {
        basis.unpack_index_to_letters(letters, out_degree, out_index);
        return letter_loop(ctx,
                           out_index,
                           out_degree,
                           letters,
                           integrator,
                           integrand,
                           op_min_deg,
                           op_max_deg);
    }


public:
    template <typename TensorOut,
              typename TensorIntegrator,
              typename TensorIntegrand>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorIntegrator const& integrator,
                               TensorIntegrand const& integrand,
                               Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const& basis = out.basis();
        Letter letter_data[Strategy::Architecture::max_depth];
        LetterSpan letters{letter_data, Strategy::Architecture::max_depth};

        const auto low_range_degree =
            std::max<Degree>(0, ctx.low_range_degree(out));


        for (Degree out_degree = out.max_degree();
             out_degree > low_range_degree;
             --out_degree) {
            const auto op_min_deg =
                std::max({Degree{0},
                          integrator.min_degree() - 1,
                          integrand.min_degree() - 1 - out_degree});
            const auto op_max_deg =
                std::min(integrator.max_degree() - 1,
                         integrand.max_degree() - 1 - out_degree);
            const auto begin = basis.start_of_degree(out_degree);
            const auto end = basis.end_of_degree(out_degree);
            for (Index i = begin + ctx.thread_rank(); i < end;
                 i += ctx.num_threads()) {
                auto acc = evaluate(ctx,
                                    out_degree,
                                    i - begin,
                                    letters,
                                    basis,
                                    integrator,
                                    integrand,
                                    op_min_deg,
                                    op_max_deg);
                out[i] = static_cast<Scalar>(beta * acc);
            }
        }

        const auto end = basis.end_of_degree(low_range_degree);
        const Index i = out.begin_index() + ctx.thread_rank();
        if (i < end) {
            const auto degree = basis.degree(i);
            const auto index = i - basis.start_of_degree(degree);
            const auto op_min_deg =
                std::max({Degree{0},
                          integrator.min_degree() - 1,
                          integrand.min_degree() - 1 - degree});
            const auto op_max_deg =
                std::min(integrator.max_degree() - 1,
                         integrand.max_degree() - 1 - degree);

            auto acc = evaluate(ctx,
                                degree,
                                index,
                                letters,
                                basis,
                                integrator,
                                integrand,
                                op_min_deg,
                                op_max_deg);
            out[i] = static_cast<Scalar>(beta * acc);
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_LMUL_HPP
