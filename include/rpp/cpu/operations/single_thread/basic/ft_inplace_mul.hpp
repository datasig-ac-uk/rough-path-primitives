#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_INPLACE_MUL_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_INPLACE_MUL_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/basic/ft_inplace_mul.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class FTInplaceMul<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    template <typename Basis>
    static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template <typename TensorLhs, typename TensorRhs>
    void operator()(Context const& ctx, TensorLhs& lhs, TensorRhs const& rhs, Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorLhs::value_type;
        ignore_unused(ctx);

        const auto out_min_degree = std::max(Degree{1}, lhs.min_degree());
        for (Degree out_degree = lhs.max_degree(); out_degree >= out_min_degree; --out_degree) {
            auto out_frag = lhs.degree_view(out_degree);

            if (rhs.min_degree() == 0) {
                for (Index i = 0; i < out_frag.size(); ++i) {
                    out_frag[i] = static_cast<Scalar>(beta * Accum{out_frag[i]} * Accum{rhs[0]});
                }
            } else {
                for (Index i = 0; i < out_frag.size(); ++i) {
                    out_frag[i] = Scalar{0};
                }
            }

            const auto lhs_degree_min = std::max(lhs.min_degree(), static_cast<Degree>(out_degree - rhs.max_degree()));
            const auto lhs_degree_max = static_cast<Degree>(out_degree - std::max(Degree{1}, rhs.min_degree()));
            if (lhs_degree_max < lhs_degree_min) {
                continue;
            }

            for (Degree lhs_degree = lhs_degree_max; lhs_degree >= lhs_degree_min; --lhs_degree) {
                const auto rhs_degree = static_cast<Degree>(out_degree - lhs_degree);
                if (!lhs.has_degree(lhs_degree) || !rhs.has_degree(rhs_degree)) {
                    continue;
                }

                auto lhs_frag = lhs.degree_view(lhs_degree);
                auto rhs_frag = rhs.degree_view(rhs_degree);

                for (Index i = 0; i < lhs_frag.size(); ++i) {
                    for (Index j = 0; j < rhs_frag.size(); ++j) {
                        const auto out_idx = i * rhs_frag.size() + j;
                        const Accum acc = Accum{out_frag[out_idx]}
                            + beta * Accum{lhs_frag[i]} * Accum{rhs_frag[j]};
                        out_frag[out_idx] = static_cast<Scalar>(acc);
                    }
                }
            }
        }

        if (lhs.min_degree() == 0 && rhs.min_degree() == 0) {
            lhs[0] = static_cast<Scalar>(beta * Accum{lhs[0]} * Accum{rhs[0]});
        } else if (lhs.min_degree() == 0) {
            lhs[0] = Scalar{0};
        }
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, typename Architecture>
void ft_inplace_mul_kernel(
    const BatchLhs batch_lhs,
    const BatchRhs batch_rhs,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ beta = Accum_{1}
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::FTInplaceMul<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto lhs = batch_lhs.view(tensor_idx, basis);
            auto rhs = batch_rhs.view(tensor_idx, basis);
            op(ctx, lhs, rhs, beta);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_INPLACE_MUL_HPP
