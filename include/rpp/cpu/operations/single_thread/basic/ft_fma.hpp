#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_FMA_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_FMA_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/basic/ft_fma.hpp>

#include <rpp/cpu/strategies.hpp>

#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>


namespace rpp::ops {
template<typename Accum_, typename Architecture>
class FTFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture> > : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

public:
    template<typename TensorOut, typename TensorA, typename TensorB, typename TensorC>
    void operator()(
        Context const &ctx,
        TensorOut &out,
        TensorA const &a,
        TensorB const &b,
        TensorC const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) const noexcept {
        auto out_min_degree = std::max(Degree{1}, out.min_degree());

        for (Degree out_degree = out.max_degree(); out_degree >= out_min_degree; --
             out_degree) {
            auto const lhs_deg_max = std::min(b.max_degree(),
                                              out_degree - c.min_degree());
            auto const lhs_deg_min = std::max(b.min_degree(),
                                              out_degree - c.max_degree());

            auto out_frag = out.degree_view(out_degree);
            if (a.has_degree(out_degree)) {
                const auto a_frag = a.degree_view(out_degree);
                for (Index i = 0; i < out_frag.size(); ++i) {
                    out_frag[i] = alpha * a_frag[i];
                }
            } else {
                for (Index i = 0; i < out_frag.size(); ++i) {
                    out_frag[i] = Accum{0};
                }
            }

            for (Degree lhs_degree = lhs_deg_max; lhs_degree >= lhs_deg_min; --
                 lhs_degree) {
                auto const rhs_degree = out_degree - lhs_degree;

                auto lhs_frag = b.degree_view(lhs_degree);
                auto rhs_frag = c.degree_view(rhs_degree);

                for (Index i = 0; i < lhs_frag.size(); ++i) {
                    for (Index j = 0; j < rhs_frag.size(); ++j) {
                        out_frag[i * rhs_frag.size() + j] +=
                                beta * lhs_frag[i] * rhs_frag[j];
                    }
                }
            }
        }

        if (out.min_degree() == 0) {
            Accum val { 0 };
            if (a.min_degree() == 0) {
                val += alpha * Accum{a[0]};
            }
            if (b.min_degree() == 0 && c.min_degree() == 0) {
                const Accum b_val {b[0]};
                const Accum c_val {c[0]};
                val += beta * b_val * c_val;
            }
            out[0] = val;
        }
    }
};
} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchOut, typename BatchA, typename BatchB, typename BatchC, typename Basis, typename Accum_, typename Architecture>
void ft_fma_kernel(
    const BatchOut batch_out,
    const BatchA batch_a,
    const BatchB batch_b,
    const BatchC batch_c,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1},
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::FTFma<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto out = batch_out.view(tensor_idx, basis);
            auto a = batch_a.view(tensor_idx, basis);
            auto b = batch_b.view(tensor_idx, basis);
            auto c = batch_c.view(tensor_idx, basis);
            op(ctx, out, a, b, c, alpha, beta);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_FMA_HPP
