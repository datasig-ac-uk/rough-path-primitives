#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_set_identity.hpp>

#include <rpp/gpu/block/operations/linalg/vector_set_constant.hpp>
#include <rpp/gpu/block/strategy.hpp>

namespace rpp::ops {

template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class TensorSetIdentity<
    gpu::strategies::
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

private:
    using SetConstant = VectorSetConstant<Strategy>;
    SetConstant set_constant;

public:
    static constexpr bool is_implemented = true;


    template <typename Tensor>
    RPP_DEVICE void operator()(Context const& ctx,
                               Tensor& tensor,
                               Accum scalar = Accum{1}) const noexcept {
        set_constant(ctx, tensor, Accum{0});
        if (ctx.thread_rank() == 0) {
            tensor[0] = scalar;
        }
    }
};

} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_SET_IDENTITY_HPP