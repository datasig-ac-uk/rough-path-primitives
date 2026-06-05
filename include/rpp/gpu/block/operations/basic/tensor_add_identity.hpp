#ifndef RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_ADD_IDENTITY_HPP
#define RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_ADD_IDENTITY_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/views/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_add_identity.hpp>

#include <rpp/gpu/block/strategy.hpp>

namespace rpp::ops {
template <typename Accum_,
          unsigned BlockSize,
          unsigned MaxBlockSize,
          typename Architecture>
class TensorAddIdentity<
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

    static constexpr bool is_implemented = true;

    template <typename Tensor>
    RPP_DEVICE void operator()(Context const& ctx,
                               Tensor& tensor,
                               Accum scalar = Accum{1}) const noexcept {
        if (ctx.thread_rank() == 0 && tensor.has_degree(0)) {
            tensor[0] += scalar;
        }
    }
};
} // namespace rpp::ops

#endif // RPP_GPU_BLOCK_OPERATIONS_BASIC_TENSOR_ADD_IDENTITY_HPP
