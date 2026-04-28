#ifndef RPP_CPU_OPS_SINGLE_THREAD_VECTOR_INPLACE_ADD_HPP
#define RPP_CPU_OPS_SINGLE_THREAD_VECTOR_INPLACE_ADD_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/cpu/strategies.hpp>
#include <rpp/operations.hpp>
#include <rpp/utility.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture>
class VectorInplaceAdd<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>> {
    using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

public:
    template <typename LaunchConfig, typename Basis>
    static constexpr std::size_t scratch_space_size(LaunchConfig const& config, Basis const& basis) noexcept {
        ignore_unused(config, basis);
        return 0;
    }

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

#endif // RPP_CPU_OPS_SINGLE_THREAD_VECTOR_INPLACE_ADD_HPP
