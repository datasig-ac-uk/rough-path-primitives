#ifndef RPP_CPU_OPS_SINGLE_THREAD_ST_INPLACE_FMA_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_ST_INPLACE_FMA_HPP

#include <algorithm>
#include <array>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/basic/st_inplace_fma.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class STInplaceFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

    template <typename Basis, typename TensorB, typename TensorC>
    static Accum shuffle_product_coefficient(
        Basis const& basis,
        TensorB const& b,
        TensorC const& c,
        Degree degree,
        Index index
    ) noexcept {
        std::array<Letter, Strategy::Architecture::max_depth> letters{};
        basis.unpack_index_to_letters(letters, degree, index);

        Accum acc{0};
        const auto mask_count = static_cast<Bitmask>(Bitmask{1} << degree);
        for (Bitmask mask{0}; mask < mask_count; ++mask) {
            Degree lhs_degree{0};
            Index lhs_idx{0};
            Degree rhs_degree{0};
            Index rhs_idx{0};
            basis.pack_masked_index(
                letters,
                degree,
                mask,
                lhs_degree,
                lhs_idx,
                rhs_degree,
                rhs_idx
            );

            if (b.has_degree(lhs_degree) && c.has_degree(rhs_degree)) {
                acc += Accum{b.degree_view(lhs_degree)[lhs_idx]} * Accum{c.degree_view(rhs_degree)[rhs_idx]};
            }
        }
        return acc;
    }

public:
    template <typename TensorA, typename TensorB, typename TensorC>
    void operator()(
        Context const& ctx,
        TensorA& a,
        TensorB const& b,
        TensorC const& c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) const noexcept {
        using Scalar = typename TensorA::value_type;
        ignore_unused(ctx);

        auto const& basis = a.basis();

        if (a.min_degree() == 0) {
            Accum value = alpha * Accum{a[0]};
            if (b.has_degree(Degree{0}) && c.has_degree(Degree{0})) {
                value += beta * Accum{b[0]} * Accum{c[0]};
            }
            a[0] = static_cast<Scalar>(value);
        }

        const auto min_degree = std::max(Degree{1}, a.min_degree());
        for (Degree degree = min_degree; degree <= a.max_degree(); ++degree) {
            auto a_level = a.degree_view(degree);
            for (Index i = 0; i < a_level.size(); ++i) {
                const Accum value = alpha * Accum{a_level[i]}
                    + beta * shuffle_product_coefficient(basis, b, c, degree, i);
                a_level[i] = static_cast<Scalar>(value);
            }
        }
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchA, typename BatchB, typename BatchC, typename Basis, typename Accum_, typename Architecture>
void st_inplace_fma_kernel(
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
    using Op = ops::STInplaceFma<Strategy>;

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

#endif // RPP_CPU_OPS_SINGLE_THREAD_ST_INPLACE_FMA_HPP
