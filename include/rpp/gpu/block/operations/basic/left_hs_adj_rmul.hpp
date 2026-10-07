#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP

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

        if (out.min_degree() == 0 && ctx.thread_rank() == 0) {
            out[0] = Scalar{0};
        }

        const auto begin_index = std::max<Index>(1, out.begin_index());
        for (Index i = begin_index + ctx.thread_rank(); i < out.end_index();
             i += ctx.num_threads()) {
            const auto out_degree = basis.degree(i);
            const auto trailing_degree = out_degree - 1;
            const auto relative_index = i - basis.start_of_degree(out_degree);
            const auto trailing_size = basis.size_of_degree(trailing_degree);
            const auto prefix = relative_index / trailing_size;
            const auto trailing_index = relative_index % trailing_size;

            const auto op_min_degree = std::max(
                integrand.min_degree(), integrator.min_degree() - out_degree);
            const auto op_max_degree = std::min(
                integrand.max_degree(), integrator.max_degree() - out_degree);

            common::detail::PrefixLetterGetter<TensorIntegrator> arg_getter{
                prefix};
            common::detail::DefaultGetter<TensorIntegrand> op_getter{};
            basis.unpack_index_to_letters(
                letters, trailing_degree, trailing_index);
            auto const acc = common::shuffle_adjoint_op_loop(ctx,
                                                             trailing_index,
                                                             trailing_degree,
                                                             letters,
                                                             integrand,
                                                             integrator,
                                                             op_min_degree,
                                                             op_max_degree,
                                                             arg_getter,
                                                             op_getter);
            out[i] = static_cast<Scalar>(beta * acc);
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
