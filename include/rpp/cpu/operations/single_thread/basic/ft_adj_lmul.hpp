#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_ADJ_LMUL_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_ADJ_LMUL_HPP

#include <cstddef>

#include <rpp/cpu/operations/single_thread/strategy.hpp>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/ft_adj_lmul.hpp>

namespace rpp::ops {
template<typename Accum_, typename Architecture>
class FTAdjLMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture> > : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

public:
    static constexpr bool is_implemented = true;


    template<typename TensorOut, typename TensorOp, typename TensorArg>
    void operator()(Context const &ctx, TensorOut &out, TensorOp const &op, TensorArg const &arg) const noexcept {
        /*
         * There are several choices for strategy here. There is a relationship
         * between the output degre (odeg), the operator degree (opdeg) and the
         * argument degree (adeg) given by
         *
         *   adeg = opdeg + odeg.
         *
         * This means we have a choice of traversal pattern over each of the
         * arguments in terms of which combination of degrees we access and in what
         * order these combinations appear. For instance, we might major the output
         * degree, in which case we compute all of the terms of a given odeg before
         * moving on to the next. This gives some notion of locality in the output,
         * but not in either of the input arguments. Alternatively, we might major
         * the operator degree or the argument degree. There are probably scenarios
         * where each choice of major is optimal, but I don't yet understand the
         * relationships. However, it is clear that the argument degree will usually
         * be the largest at play at any given time, so it is reasonable to major
         * the argument degree.
         *
         * This, of course, is ignoring the ability to tile the operation in the
         * same way as we perform a tiled fma or antipode operation. This is a
         * future improvement because it requires the same kind of layout framework
         * as those operations do.
         */

        const auto arg_max_degree = std::min(arg.max_degree() - op.min_degree(), out.max_degree());
        auto arg_min_degree = std::max(arg.min_degree() - op.max_degree(), out.min_degree());

        for (Degree arg_degree = arg_max_degree; arg_degree >= arg_min_degree; --
             arg_degree) {
            auto out_min_degree = std::max(arg_degree - op.max_degree(), out.min_degree());
            auto out_max_degree = std::min(arg_degree - op.min_degree(), out.max_degree());

            auto arg_frag = arg.degree_view(arg_degree);


            for (Degree out_degree = out_max_degree;
                 out_degree >= out_min_degree;
                 --out_degree) {
                const auto op_degree = arg_degree - out_degree;

                auto op_frag = op.degree_view(op_degree);
                auto out_frag = out.degree_view(out_degree);


                // ReSharper disable CppDFANullDereference
                for (Index op_idx = 0; op_idx < op_frag.size(); ++op_idx) {
                    const auto op_offset = op_idx * out_frag.size();
                    const auto &op_val = op_frag[op_idx];

                    for (Index i = 0; i < out_frag.size(); ++i) {
                        out_frag[i] += op_val * arg_frag[i + op_offset];
                    }
                }
                // ReSharper restore CppDFANullDereference
            }
        }
    }
};
} // namespace rpp::ops

#endif // RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_FT_ADJ_LMUL_HPP
