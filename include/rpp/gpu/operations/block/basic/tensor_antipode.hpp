#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_ANTIPODE_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_ANTIPODE_HPP

#include <rpp/config.h>
#include <rpp/utility.hpp>
#include <rpp/dense/batch.hpp>


#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/operations/block/basic/tensor_generalised_antipode.hpp>

namespace rpp::gpu::block {

template <typename BatchOut, typename BatchArg, typename Basis, typename Accum_, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void tensor_antipode_kernel(
    const BatchOut batch_out,
    const BatchArg batch_arg,
    const Basis basis,
    const strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::BlockStrategy<Accum_, MaxBlockSize, Architecture>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    ops::TensorAntipode<Strategy> op;

    auto out = batch_out.view(my_index, basis);
    auto arg = batch_arg.view(my_index, basis);
    op(ctx, out, arg);
}

} // namespace rpp::gpu::block

#endif // RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_ANTIPODE_HPP
