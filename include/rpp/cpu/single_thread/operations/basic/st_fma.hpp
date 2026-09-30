#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_FMA_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_FMA_HPP

#include <algorithm>
#include <array>
#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/st_fma.hpp>
#include <rpp/operations/implementation/word_shuffle_product.hpp>

#include <rpp/cpu/single_thread/strategy.hpp>
namespace rpp::ops {

template <typename Accum_, typename Architecture>
class STFma<cpu::strategies::SingleThreadStrategy<Accum_, Architecture>>
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

    template <typename TensorOut,
              typename TensorA,
              typename TensorB,
              typename TensorC>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorA const& a,
                    TensorB const& b,
                    TensorC const& c,
                    Accum alpha = Accum{1},
                    Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        ignore_unused(ctx);

        auto const& basis = out.basis();

        if (out.min_degree() == 0) {
            Accum value{0};
            if (a.has_degree(Degree{0})) {
                value += alpha * Accum{a[0]};
            }
            if (b.has_degree(Degree{0}) && c.has_degree(Degree{0})) {
                value += beta * Accum{b[0]} * Accum{c[0]};
            }
            out[0] = static_cast<Scalar>(value);
        }

        const auto min_degree = std::max(Degree{1}, out.min_degree());
        for (Degree degree = min_degree; degree <= out.max_degree(); ++degree) {
            auto out_level = out.degree_view(degree);
            for (Index i = 0; i < out_level.size(); ++i) {
                Accum value{0};
                if (a.has_degree(degree)) {
                    value += alpha * Accum{a.degree_view(degree)[i]};
                }
                auto word_product =
                    common::word_shuffle_product(ctx, i, degree, b, c);
                value += beta * word_product;
                out_level[i] = static_cast<Scalar>(value);
            }
        }
    }
};

} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_ST_FMA_HPP
