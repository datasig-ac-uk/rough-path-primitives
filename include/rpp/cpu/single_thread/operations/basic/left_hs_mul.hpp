//
// Created by sam on 30/09/2026.
//

#ifndef RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
#define RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_MUL_HPP

// IWYU pragma: always_keep

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/left_hs_mul.hpp>
#include <rpp/operations/implementation/word_half_shuffle_product.hpp>
#include <rpp/views/batch.hpp>


#include <rpp/cpu/single_thread/strategy.hpp>


namespace rpp::ops {

template <typename AccumT, typename ArchiectureT>
class LeftHSMul<cpu::strategies::SingleThreadStrategy<AccumT, ArchiectureT>>
    : public BaseOperation<
          cpu::strategies::SingleThreadStrategy<AccumT, ArchiectureT>> {
    using Strategy =
        cpu::strategies::SingleThreadStrategy<AccumT, ArchiectureT>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

public:
    static constexpr bool is_implemented = true;

    template <typename TensorOut,
              typename TensorIntegrator,
              typename TensorIntegrand>
    void operator()(Context const& ctx,
                    TensorOut& out,
                    TensorIntegrator const& integrator,
                    TensorIntegrand const& integrand,
                    Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;

        auto const& basis = out.basis();


        if (out.min_degree() == 0) {
            out[0] = Scalar{0};
        }


        const auto min_degree = std::max<Degree>(1, out.min_degree());
        const auto alphabet_size = static_cast<Index>(basis.width);
        for (Degree degree = min_degree; degree <= out.max_degree(); ++degree) {
            const auto trailing_degree = degree - 1;
            const auto trailing_size = basis.size_of_degree(trailing_degree);
            auto out_level = out.degree_view(degree);

            for (Index prefix_letter = 0; prefix_letter < alphabet_size;
                 ++prefix_letter) {
                for (Index i = 0; i < trailing_size; ++i) {
                    const auto word_product =
                        common::word_left_half_shuffle_product(ctx,
                                                               prefix_letter,
                                                               i,
                                                               trailing_degree,
                                                               integrator,
                                                               integrand);
                    out_level[prefix_letter * trailing_size + i] =
                        static_cast<Scalar>(beta * word_product);
                }
            }
        }
    }
};


} // namespace rpp::ops

#endif // RPP_CPU_SINGLE_THREAD_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
