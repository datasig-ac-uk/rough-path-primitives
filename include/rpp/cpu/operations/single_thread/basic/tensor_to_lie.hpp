#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_TENSOR_TO_LIE_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_TENSOR_TO_LIE_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/tensor_to_lie.hpp>

#include <rpp/cpu/strategies.hpp>
#include <rpp/cpu/operations/single_thread/basic/sparse_matrix_vector_product.hpp>

namespace rpp::cpu::single_thread {

template <typename LieBatchOut, typename TensorBatchIn, typename Matrix, typename Accum, typename Architecture>
void tensor_to_lie(
    const LieBatchOut lie_batch_out,
    const TensorBatchIn tensor_batch_in,
    const Matrix matrix,
    const strategies::SingleThreadStrategy<Accum, Architecture> strategy,
    typename Architecture::Index n_tensors
    ) {
    using Strategy = strategies::SingleThreadStrategy<Accum, Architecture>;

    static constexpr auto impl_type = sparse::matrix_format_v<Matrix> == sparse::CSRMatrix
        ? ops::T2LImplementationType::CSRSparseMatrix : ops::T2LImplementationType::CSCSparseMatrix;

    using Op = ops::TensorToLie<Strategy, impl_type>;

    detail::apply_batch<Op>(
        lie_batch_out.basis(),
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index idx) {
            auto out = lie_batch_out.view(idx);
            auto in = tensor_batch_in.view(idx);
            op(ctx, out, in, matrix);
        }
    );

}

} // namespace rpp::cpu::single_thread

#endif //RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_TENSOR_TO_LIE_HPP
