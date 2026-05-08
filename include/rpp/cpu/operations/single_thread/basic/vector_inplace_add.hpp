#ifndef RPP_CPU_OPS_SINGLE_THREAD_VECTOR_INPLACE_ADD_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_VECTOR_INPLACE_ADD_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/dense/batch.hpp>

#include <rpp/operations/basic/vector_inplace_add.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/detail/batch_wrapper.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class VectorInplaceAdd<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> : public BaseOperation<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    template <typename VectorLhs, typename VectorRhs>
    void operator()(Context const& ctx, VectorLhs& lhs, VectorRhs const& rhs, Accum alpha = Accum{1}) const noexcept {
        using Scalar = typename VectorLhs::value_type;
        ignore_unused(ctx);

        auto const& basis = lhs.basis();
        const auto min_degree = std::max(lhs.min_degree(), rhs.min_degree());
        const auto max_degree = std::min(lhs.max_degree(), rhs.max_degree());
        if (max_degree < min_degree) {
            return;
        }

        const auto begin = basis.start_of_degree(min_degree);
        const auto size = basis.end_of_degree(max_degree) - begin;
        auto lhs_it = lhs.data() + begin;
        auto rhs_it = rhs.data() + begin;


        for (Index i=0; i < size; ++i) {
            const Accum lhs_val { lhs_it[i] };
            const Accum rhs_val { rhs_it[i] };
            const Accum result = lhs_val + alpha* rhs_val;
            lhs_it[i] = static_cast<Scalar>(result);
        }
    }
};

} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename BatchLhs, typename BatchRhs, typename Basis, typename Accum_, typename Architecture>
void vector_inplace_add_kernel(
    const BatchLhs batch_lhs,
    const BatchRhs batch_rhs,
    const Basis basis,
    const strategies::SingleThreadStrategy<Accum_, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ alpha = Accum_{1}
) {
    using Strategy = strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Op = ops::VectorInplaceAdd<Strategy>;

    detail::apply_batch<Op>(
        basis,
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto lhs = batch_lhs.view(tensor_idx, basis);
            auto rhs = batch_rhs.view(tensor_idx, basis);
            op(ctx, lhs, rhs, alpha);
        }
    );
}

} // namespace rpp::cpu::single_thread

#endif // RPP_CPU_OPS_SINGLE_THREAD_VECTOR_INPLACE_ADD_HPP
