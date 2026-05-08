#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_SET_IDENTITY_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_SET_IDENTITY_HPP

#include <rpp/config.h>
#include <rpp/dense/batch.hpp>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_set_identity.hpp>

#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/operations/block/basic/vector_set_constant.hpp>

namespace rpp::ops {

template <typename Accum_, unsigned BlockSize, typename Architecture>
class TensorSetIdentity<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> : public BaseOperation<gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>> {
public:
    using Strategy = gpu::strategies::BlockStrategy<Accum_, BlockSize, Architecture>;
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;


private:
    using SetConstant = VectorSetConstant<Strategy>;
    SetConstant set_constant;
public:



    template <typename Tensor>
    RPP_DEVICE void operator()(Context const& ctx, Tensor& tensor, Accum scalar = Accum{1}) const noexcept {
        set_constant(ctx, tensor, Accum{0});
        if (ctx.thread_rank() == 0) {
            tensor[0] = scalar;
        }
    }
};

} // namespace rpp::ops

namespace rpp::gpu::block {

template <typename BatchTensor, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void tensor_set_identity_kernel(
    const BatchTensor batch_tensor,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors,
    Accum_ scalar = Accum_{1}
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::TensorSetIdentity<Strategy> op;

    auto tensor = batch_tensor.view(my_index, basis);
    op(ctx, tensor, scalar);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_SET_IDENTITY_HPP
