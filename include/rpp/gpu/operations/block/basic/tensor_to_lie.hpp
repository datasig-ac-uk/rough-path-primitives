#ifndef RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_TO_LIE_HPP
#define RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_TO_LIE_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/base_operation.hpp>
#include <rpp/operations/basic/tensor_to_lie.hpp>

#include <rpp/gpu/strategies.hpp>
#include <rpp/gpu/operations/block/basic/sparse_matrix_vector.hpp>

namespace rpp::gpu::block {

template <typename LieBatchOut, typename TensorBatchIn, typename Matrix, typename Accum, unsigned MaxBlockSize, typename Architecture>
RPP_KERNEL void tensor_to_lie_kernel(
    const LieBatchOut lie_batch_out,
    const TensorBatchIn tensor_batch_in,
    const Matrix matrix,
    const strategies::BlockStrategy<Accum, MaxBlockSize, Architecture> strategy,
    typename Architecture::Index n_tensors
) {
    using Strategy = strategies::BlockStrategy<Accum, MaxBlockSize, Architecture>;

    static constexpr auto impl_type = sparse::matrix_format_v<Matrix> == sparse::MatrixFormat::CSR
        ? ops::T2LImplementationType::CSRSparseMatrix : ops::T2LImplementationType::CSCSparseMatrix;

    using Op = ops::TensorToLie<Strategy, impl_type>;

    extern __shared__ std::byte smem_bytes[];

    const auto ctx = strategy.make_context(smem_bytes);
    const auto my_index = strategy.object_index(blockIdx.x, threadIdx.x);
    if (my_index >= n_tensors) { return; }

    Op op;

    auto out = lie_batch_out.view(my_index, tensor_batch_in.basis());
    auto in = tensor_batch_in.view(my_index, tensor_batch_in.basis());
    op(ctx, out, in, matrix);
}

} // namespace rpp::gpu::block

#endif //RPP_GPU_OPERATIONS_BLOCK_BASIC_TENSOR_TO_LIE_HPP
