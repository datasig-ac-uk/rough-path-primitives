#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP

// IWYU pragma: always_keep

#include <algorithm>

#include <rpp/gpu/block/strategy.hpp>
#include <rpp/operations/basic/left_hs_adj_rmul.hpp>
#include <rpp/operations/implementation/shuffle_adjoint_op_loop.hpp>
#include <rpp/operations/implementation/word_half_shuffle_product.hpp>

namespace rpp::ops {

template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class LeftHSAdjRMul<
    gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;
    using Letter = typename Strategy::Letter;
    using LetterSpan = Span<Letter>;

    static constexpr bool is_implemented = true;

private:
    template <typename Basis,
              typename TensorIntegrator,
              typename TensorIntegrand>
    RPP_DEVICE static Accum evaluate(Context const& ctx,
                                     Degree out_degree,
                                     Index relative_index,
                                     LetterSpan letters,
                                     Basis const& basis,
                                     TensorIntegrator const& integrator,
                                     TensorIntegrand const& integrand,
                                     Degree op_min_degree,
                                     Degree op_max_degree) noexcept {
        const auto trailing_degree = out_degree - 1;
        const auto trailing_size = basis.size_of_degree(trailing_degree);
        const auto prefix = relative_index / trailing_size;
        const auto trailing_index = relative_index % trailing_size;


        common::detail::PrefixLetterGetter<TensorIntegrator> arg_getter{prefix};
        common::detail::DefaultGetter<TensorIntegrand> op_getter{};
        basis.unpack_index_to_letters(letters, trailing_degree, trailing_index);
        return common::shuffle_adjoint_op_loop(ctx,
                                               trailing_index,
                                               trailing_degree,
                                               letters,
                                               integrand,
                                               integrator,
                                               op_min_degree,
                                               op_max_degree,
                                               arg_getter,
                                               op_getter);
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
            const auto op_min_degree = std::max(
                integrand.min_degree(), integrator.min_degree() - out_degree);
            const auto op_max_degree = std::min(
                integrand.max_degree(), integrator.max_degree() - out_degree);
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
                                    op_min_degree,
                                    op_max_degree);

                out[i] = static_cast<Scalar>(beta * acc);
            }
        }

        const auto end = basis.end_of_degree(low_range_degree);
        const Index i = out.begin_index() + ctx.thread_rank();
        if (i < end) {
            Accum acc{0};

            if (i > 0) {
                const auto degree = basis.degree(i);
                const auto index = i - basis.start_of_degree(degree);
                const auto op_min_degree = std::max(
                    integrand.min_degree(), integrator.min_degree() - degree);
                const auto op_max_degree = std::min(
                    integrand.max_degree(), integrator.max_degree() - degree);

                acc = evaluate(ctx,
                               degree,
                               index,
                               letters,
                               basis,
                               integrator,
                               integrand,
                               op_min_degree,
                               op_max_degree);
            }

            out[i] = static_cast<Scalar>(beta * acc);
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
