#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_INPLACE_FMA_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_INPLACE_FMA_HPP

// IWYU pragma: always_keep

#include <algorithm>
#include <array>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/st_inplace_fma.hpp>
#include <rpp/operations/implementation/word_shuffle_product.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
namespace rpp::ops {

template <typename Accum_, typename Architecture>
class STInplaceFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
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

public:
    static constexpr bool is_implemented = true;

    template <typename TensorA, typename TensorB, typename TensorC>
    void operator()(Context const& ctx,
                    TensorA& a,
                    TensorB const& b,
                    TensorC const& c,
                    Accum alpha = Accum{1},
                    Accum beta = Accum{1}) const noexcept {
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
                auto word_product =
                    common::word_shuffle_product(ctx, i, degree, b, c);
                const Accum value = alpha * a_level[i] + beta * word_product;
                a_level[i] = static_cast<Scalar>(value);
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_INPLACE_FMA_HPP
