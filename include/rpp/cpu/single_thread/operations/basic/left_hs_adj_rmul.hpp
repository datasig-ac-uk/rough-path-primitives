#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP

// IWYU pragma: always_keep

#include <rpp/config.h>

#include <rpp/support/span.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
#include <rpp/operations/basic/left_hs_adj_rmul.hpp>
#include <rpp/operations/implementation/word_half_shuffle_product.hpp>

#include <rpp/operations/implementation/shuffle_adjoint_op_loop.hpp>


namespace rpp::ops {

template <typename Accum_, typename Architecture>
class LeftHSAdjRMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
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

    static constexpr bool is_implemented = true;

    template <typename TensorOut,
              typename TensorIntegrator,
              typename TensorIntegrand>
    void operator()(Context const& ctx,
                    TensorOut const& out,
                    TensorIntegrator const& integrator,
                    TensorIntegrand const& integrand) const noexcept {
        using Scalar = typename TensorOut::value_type;

        auto const& basis = out.basis();

        Letter letter_data[Strategy::Architecture::max_depth];
        LetterSpan letters{letter_data, Strategy::Architecture::max_depth};

        const auto width = static_cast<Index>(basis.width);
        const auto out_min_deg = out.min_degree();
        const auto out_max_deg = out.max_degree();

        if (out_min_deg == 0) {
            out[0] = Scalar{0};
        }

        for (Degree out_degree = std::max<Degree>(1, out_min_deg);
             out_degree <= out_max_deg;
             ++out_degree) {
            const auto effective_out_deg = out_degree - 1;
            auto out_level = out.degree_view(out_degree);

            const auto trailing_size = basis.size_of_degree(effective_out_deg);


            const auto op_min_deg = std::max(
                integrand.min_degree(), integrator.min_degree() - out_degree);
            const auto op_max_deg = std::min(
                integrand.max_degree(), integrator.max_degree() - out_degree);

            for (Index prefix = 0; prefix < width; ++prefix) {

                common::detail::PrefixLetterGetter<TensorIntegrator> arg_getter{
                    prefix};
                common::detail::DefaultGetter<TensorIntegrand> op_getter{};

                for (Index i = 0; i < trailing_size; ++i) {
                    basis.unpack_index_to_letters(
                        letters, effective_out_deg, i);

                    auto acc =
                        common::shuffle_adjoint_op_loop(ctx,
                                                        i,
                                                        effective_out_deg,
                                                        letters,
                                                        integrand,
                                                        integrator,
                                                        op_min_deg,
                                                        op_max_deg,
                                                        arg_getter,
                                                        op_getter);


                    out_level[prefix * trailing_size + i] =
                        static_cast<Scalar>(acc);
                }
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_ADJ_RMUL_HPP
