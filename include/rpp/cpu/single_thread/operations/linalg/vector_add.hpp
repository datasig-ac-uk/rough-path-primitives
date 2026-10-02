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
        using Scalar = typename VectorOut::Scalar;
        ignore_unused(ctx);

        auto const& basis = out.basis();

        for (auto degree = out.min_degree(); degree <= out.max_degree();
             ++degree) {
            const bool has_lhs = lhs.has_degree(degree);
            const bool has_rhs = rhs.has_degree(degree);

            for (Index i = basis.start_of_degree(degree);
                 i < basis.end_of_degree(degree);
                 ++i) {
                Accum val{0};
                if (has_lhs) {
                    val += alpha * Accum{lhs[i]};
                }
                if (has_rhs) {
                    val += beta * Accum{rhs[i]};
                }
                out[i] = static_cast<Scalar>(val);
            }
        }
    }
};


} // namespace rpp::ops


#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_LINALG_VECTOR_ADD_HPP
