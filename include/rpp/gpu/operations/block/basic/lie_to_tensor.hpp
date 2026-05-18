#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_LIE_TO_TENSOR_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_LIE_TO_TENSOR_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/lie_to_tensor.hpp>

#include <rpp/gpu/operations/block/strategy.hpp>
#include <rpp/gpu/operations/block/linalg/sparse_matrix_vector.hpp>

namespace rpp::gpu::block {
template<typename TensorBatchOut, typename LieBatchIn, typename Matrix, typename Accum, unsigned BlockSize, unsigned
    MaxBlockSize, typename Architecture>
RPP_KERNEL void lie_to_tensor_kernel(
    const TensorBatchOut tensor_batch_out,
    const LieBatchIn lie_batch_in,
    const Matrix matrix,
    const strategies::BlockStrategy<Accum, BlockSize, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::BlockStrategy<Accum, BlockSize, MaxBlockSize, Architecture>;

    static constexpr auto impl_type = sparse::matrix_format_v<Matrix> == sparse::MatrixFormat::CSR
                                          ? ops::L2TImplementationType::CSRSparseMatrix
                                          : ops::L2TImplementationType::CSCSparseMatrix;

    using Op = ops::LieToTensor<Strategy, impl_type>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    Op op;

    auto out = tensor_batch_out.view(my_index, lie_batch_in.basis());
    auto in = lie_batch_in.view(my_index, lie_batch_in.basis());
    op(ctx, out, in, matrix);
}
} // namespace rpp::gpu::block

#endif //RPP_GPU_OPERATIONS_BLOCK_BASIC_LIE_TO_TENSOR_HPP
