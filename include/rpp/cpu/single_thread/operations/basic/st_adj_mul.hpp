#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_ADJ_MUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_ADJ_MUL_HPP

// IWYU pragma: always_keep

#include <algorithm>
#include <array>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/support/span.hpp>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/st_adj_mul.hpp>

#include <rpp/operations/implementation/shuffle_adjoint_op_loop.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class STAdjMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;
    using LetterSpan = Span<Letter>;


public:
    static constexpr bool is_implemented = true;

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const& ctx RPP_MAYBE_UNUSED,
                    TensorOut& out,
                    TensorOp const& op,
                    TensorArg const& arg) const noexcept {
        using Scalar = typename TensorOut::value_type;

        auto const& basis = out.basis();
        std::array<Letter, Strategy::Architecture::max_depth> letters{};
        LetterSpan letter_span{letters.data(),
                               Strategy::Architecture::max_depth};


        for (Degree out_degree = out.min_degree();
             out_degree <= out.max_degree();
             ++out_degree) {
            auto out_level = out.degree_view(out_degree);
            const auto op_min_deg =
                std::max(op.min_degree(), arg.min_degree() - out_degree);
            const auto op_max_deg =
                std::min(op.max_degree(), arg.max_degree() - out_degree);

            for (Index out_index = 0; out_index < out_level.size();
                 ++out_index) {
                LetterSpan out_letters{letters.data(),
                                       static_cast<size_t>(out_degree)};
                basis.unpack_index_to_letters(
                    out_letters, out_degree, out_index);

                auto acc = common::shuffle_adjoint_op_loop(ctx,
                                                           out_index,
                                                           out_degree,
                                                           letter_span,
                                                           op,
                                                           arg,
                                                           op_min_deg,
                                                           op_max_deg);


                out_level[out_index] = static_cast<Scalar>(acc);
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_ADJ_MUL_HPP
