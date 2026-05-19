#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_INPLACE_FMA_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_INPLACE_FMA_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/st_inplace_fma.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>
#include <rpp/gpu/operations/block/basic/detail/st_multiply.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class STInplaceFma<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > : public
        BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Index = typename Strategy::Index;

    static constexpr bool is_implemented = true;

    template<typename TensorA, typename TensorB, typename TensorC>
    RPP_DEVICE void operator()(
        Context const &ctx,
        TensorA &a,
        TensorB const &b,
        TensorC const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) const noexcept {
        using Scalar = typename TensorA::value_type;
        auto const &basis = a.basis();
        for (Index elt_idx = ctx.thread_rank(); elt_idx < a.size(); elt_idx += ctx.num_threads()) {
            auto acc = gpu::block::st_multiply_loop(ctx, elt_idx, basis, b, c);
            acc *= beta;
            a[elt_idx] = static_cast<Scalar>(alpha * Accum{a[elt_idx]} + acc);
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_ST_INPLACE_FMA_HPP
