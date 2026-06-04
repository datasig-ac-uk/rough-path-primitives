#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_ADD_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_ADD_HPP

#include <algorithm>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/linalg/vector_add.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>

namespace rpp::ops {

template <typename Accum_, typename Architecture_>
class VectorAdd<cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;


public:
    static constexpr bool is_implemented = true;

    using Context = typename Strategy::Context;

    template <typename VectorOut, typename VectorLhs, typename VectorRhs>
    RPP_HOST_DEVICE void operator()(Context const& ctx,
                                    VectorOut& out,
                                    VectorLhs const& lhs,
                                    VectorRhs const& rhs,
                                    Accum alpha = Accum{1},
                                    Accum beta = Accum{1}) const noexcept {
        ignore_unused(ctx);

        const auto min_deg =
            std::max({out.min_degree(), lhs.min_degree(), rhs.min_degree()});
        const auto max_deg =
            std::min({out.max_degree(), lhs.max_degree(), rhs.max_degree()});
        if (max_deg < min_deg) {
            return;
        }

        auto const& basis = out.basis();

        const auto begin = basis.start_of_degree(min_deg);
        const auto end = basis.end_of_degree(max_deg);

        for (auto i = begin; i < end; ++i) {
            const Accum left_val{lhs[i]};
            const Accum right_val{rhs[i]};

            out[i] = alpha * left_val + beta * right_val;
        }
    }
};


} // namespace rpp::ops


#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_ADD_HPP
