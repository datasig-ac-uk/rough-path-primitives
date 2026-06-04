#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_FMA_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_FMA_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_fma.hpp>

#include <rpp/gpu/block/operations/basic/detail/st_multiply.hpp>
#include <rpp/gpu/block/strategy.hpp>


namespace rpp::ops {
template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class STFma<gpu::strategies::
                BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>>
    : public BaseOperation<
          gpu::strategies::
              BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::
        BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    static constexpr bool is_implemented = true;

    template <typename TensorOut,
              typename TensorA,
              typename TensorB,
              typename TensorC>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorA const& a,
                               TensorB const& b,
                               TensorC const& c,
                               Accum alpha = Accum{1},
                               Accum beta = Accum{1}) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const& basis = out.basis();
        for (Index elt_idx = out.begin_index() + ctx.thread_rank();
             elt_idx < out.end_index();
             elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            auto acc = gpu::block::st_multiply_loop_with_degree(
                ctx, elt_idx, degree, basis, b, c);
            acc *= beta;
            Accum a_val{0};
            if (a.has_degree(degree)) {
                a_val = Accum{a[elt_idx]};
            }
            out[elt_idx] = static_cast<Scalar>(alpha * a_val + acc);
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_ST_FMA_HPP
