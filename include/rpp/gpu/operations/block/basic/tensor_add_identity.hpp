#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_ADD_IDENTITY_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_ADD_IDENTITY_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/dense/batch.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_add_identity.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>

namespace rpp::ops {
template<typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
class TensorAddIdentity<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > : public
        BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> > {
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;

public:
    static constexpr bool is_implemented = true;

    template<typename Tensor>
    RPP_DEVICE void operator()(Context const &ctx, Tensor &tensor, Accum scalar = Accum{1}) const noexcept {
        if (ctx.thread_rank() == 0) {
            tensor[0] += scalar;
        }
    }
};
} // namespace rpp::ops

namespace rpp::gpu::block {
template<typename BatchTensor, typename Basis, typename Accum_, unsigned BlockSize, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void tensor_add_identity_kernel(
    const BatchTensor batch_tensor,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ scalar = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, BlockSize, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::TensorAddIdentity<Strategy> op;

    auto tensor = batch_tensor.view(my_index, basis);
    op(ctx, tensor, scalar);
}
} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_ADD_IDENTITY_HPP
