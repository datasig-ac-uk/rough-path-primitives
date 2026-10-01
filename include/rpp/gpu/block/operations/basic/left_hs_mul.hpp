//
// Created by sam on 01/10/2026.
//

#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_MUL_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/views/batch.hpp>

#include <rpp/operations/basic/left_hs_mul.hpp>
#include <rpp/operations/implementation/word_half_shuffle_product.hpp>

#include <rpp/gpu/block/strategy.hpp>


namespace rpp::ops {


template <typename AccumT,
          unsigned BlockSizeV,
          unsigned MaxBlockSizeV,
          typename ArchitectureT>
class LeftHSMul<
    gpu::strategies::
        BlockStrategy<AccumT, BlockSizeV, MaxBlockSizeV, ArchitectureT>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<AccumT, BlockSizeV, MaxBlockSizeV, ArchitectureT>> {
public:
    using Strategy = gpu::strategies::
        BlockStrategy<AccumT, BlockSizeV, MaxBlockSizeV, ArchitectureT>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

    static constexpr bool is_implemented = true;

    template <typename TensorOut,
              typename TensorIntegrator,
              typename TensorIntegrand>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorIntegrand const& integrand,
                               TensorIntegrator const& integrator,
                               Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        const auto& basis = out.basis();

        const auto begin_index = std::max<Index>(1, out.begin_index());
        const auto end_index = out.end_index();

        for (Index i = begin_index + ctx.thread_rank(); i < end_index;
             i += ctx.num_threads()) {
            const auto degree = basis.degree(i);
            const auto trailing_degree = degree - 1;
            const auto relative_index = i - basis.start_of_degree(degree);
            const auto trailing_size = basis.size_of_degree(trailing_degree);
            const auto prefix_letter = relative_index / trailing_size;
            const auto trailing_index = i - prefix_letter * trailing_size;

            auto word_product =
                common::word_left_half_shuffle_product(ctx,
                                                       prefix_letter,
                                                       trailing_index,
                                                       trailing_degree,
                                                       integrator,
                                                       integrand);
            out[i] = static_cast<Scalar>(beta * word_product);
        }
    }
};


} // namespace rpp::ops


#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_LEFT_HS_MUL_HPP
