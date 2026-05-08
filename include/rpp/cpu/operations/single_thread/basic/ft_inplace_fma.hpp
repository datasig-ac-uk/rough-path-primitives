#ifndef RPP_CPU_OPS_SINGLE_THREAD_FT_INPLACE_FMA_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_FT_INPLACE_FMA_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/basic/ft_inplace_fma.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>

namespace rpp::ops {

// TODO: The inplace a <- a*b + c and a <- b*a + c variants need a different strategy

// template<typename Accum_, typename Architecture, InplaceFMAType FMAType>
// class FTInplaceFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>, FMAType> {
//     using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
//     using Context = typename Strategy::Context;
//     using Accum = typename Strategy::Accum;
//
//     using Index = typename Strategy::Index;
//     using Degree = typename Strategy::Degree;
//
// public:
//     template<typename Basis>
//     static constexpr std::size_t scratch_space_size(Strategy const &strategy, Basis const &basis) noexcept {
//         ignore_unused(strategy, basis);
//         return 0;
//     }
//
//     template<typename TensorA, typename TensorB, typename TensorC>
//     void operator()(
//         Context const &ctx,
//         TensorA &a,
//         TensorB const &b,
//         TensorC const &c,
//         Accum alpha = Accum{1},
//         Accum beta = Accum{1}
//     ) const noexcept {
//         RPP_UNREACHABLE();
//     }
//
// };


template<typename Accum_, typename Architecture>
class FTInplaceFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>, FTInplaceFMAType::AEqualsBCPlusA> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

public:
    template <typename Basis>
    static constexpr std::size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
        ignore_unused(strategy, basis);
        return 0;
    }

    template<typename TensorA, typename TensorB, typename TensorC>
    void operator()(
        Context const &ctx,
        TensorA &a,
        TensorB const &b,
        TensorC const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) const noexcept {
        ignore_unused(ctx);

        auto a_min_degree = std::max(Degree{1}, a.min_degree());

        for (Degree a_degree = a.max_degree(); a_degree >= a_min_degree; --
             a_degree) {
            auto const b_deg_max = std::min(b.max_degree(),
                                            a_degree - c.min_degree());
            auto const b_deg_min = std::max(b.min_degree(),
                                            a_degree - c.max_degree());

            auto a_frag = a.degree_view(a_degree);

            for (Index i=0; i<a_frag.size(); ++i) {
                a_frag[i] = alpha * a_frag[i];
            }

            for (Degree b_degree = b_deg_max; b_degree >= b_deg_min; --
                 b_degree) {
                auto const c_degree = a_degree - b_degree;

                auto b_frag = b.degree_view(b_degree);
                auto c_frag = c.degree_view(c_degree);


                for (Index i = 0; i < b_frag.size(); ++i) {
                    for (Index j = 0; j < c_frag.size(); ++j) {
                        a_frag[i * c_frag.size() + j] +=
                            beta * b_frag[i] * c_frag[j];
                    }
                }
                 }
             }

        if (a.min_degree() == 0) {
            a[0] = alpha * a[0];
            if (b.min_degree() == 0 && c.min_degree() == 0) {
                a[0] += beta * b[0] * c[0];
            }
        }
    }

};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <ops::FTInplaceFMAType FMAType, typename BatchA, typename BatchB, typename BatchC, typename Basis, typename Accum_, typename Architecture>
void ft_inplace_fma_kernel(
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
    using Op = ops::FTInplaceFma<Strategy, FMAType>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto a = batch_a.view(tensor_idx, basis);
            auto b = batch_b.view(tensor_idx, basis);
            auto c = batch_c.view(tensor_idx, basis);
            op(ctx, a, b, c, alpha, beta);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_FT_INPLACE_FMA_HPP
