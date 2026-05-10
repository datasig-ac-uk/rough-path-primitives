#ifndef RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_LIE_TO_TENSOR_HPP
#define RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_LIE_TO_TENSOR_HPP

#include <cstddef>

#include <rpp/config.h>
#include <rpp/utility.hpp>

#include <rpp/operations/basic/lie_to_tensor.hpp>

#include <rpp/cpu/operations/single_thread/strategy.hpp>
#include <rpp/cpu/operations/single_thread/basic/sparse_matrix_vector.hpp>


namespace rpp::ops {


// template <typename Accum_, typename Architecture_>
// class LieToTensor<cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>, L2TImplementationType::CommutatorExpansion> {
//     using Strategy = cpu::strategies::SingleThreadStrategy<Accum_, Architecture_>;
//     using Accum = typename Strategy::Accum;
//     using Index = typename Strategy::Index;
//     using Degree = typename Strategy::Degree;
//
// public:
//     using Context = typename Strategy::COntext;
//
//     template <typename Basis>
//     static constexpr size_t scratch_space_size(Strategy const& strategy, Basis const& basis) noexcept {
//         ignore_unused(strategy, basis);
//         return 0;
//     }
//
//     template <typename TensorOut, typename LieIn>
//     RPP_HOST_DEVICE
//     void operator()(Context const& ctx, TensorOut& out, LieIn const& arg) const noexcept {
//         ignore_unused(ctx);
//
//         auto const& lie_basis = arg.basis();
//         auto const& tensor_basis = out.basis();
//
//         for (Degree deg=std::max(1, arg.min_degree()); deg <= arg.max_degree(); ++deg) {
//             const auto lie_begin = lie_basis.start_of_degree(deg);
//             const auto lie_end = lie_basis.end_of_degree(deg);
//             const auto tensor_begin = lie_basis.start_of_degree(deg);
//             const auto tensor_end = lie_basis.end_of_degree(deg);
//
//
//
//         }
//     }
// };


} // namespace rpp::ops

namespace rpp::cpu::single_thread {

template <typename TensorBatchOut, typename LieBatchIn, typename Matrix, typename Accum, typename Architecture>
void lie_to_tensor(
    const TensorBatchOut tensor_batch_out,
    const LieBatchIn lie_batch_in,
    const Matrix matrix,
    const strategies::SingleThreadStrategy<Accum, Architecture> strategy,
    typename Architecture::Index n_tensors) noexcept {

    static constexpr auto impl_type = sparse::matrix_format_v<Matrix> == sparse::CSRMatrix
        ? ops::L2TImplementationType::CSRSparseMatrix : ops::L2TImplementationType::CSCSparseMatrix;


    using Strategy = strategies::SingleThreadStrategy<Accum, Architecture>;
    using Op = ops::LieToTensor<Strategy, impl_type>;

    detail::apply_batch<Op>(
        lie_batch_in.basis(),
        strategy,
        n_tensors,
        [&](Op const& op, typename Strategy::Context const& ctx, typename Strategy::Index tensor_idx) {
            auto tensor = tensor_batch_out.view(tensor_idx, lie_batch_in.basis());
            auto lie = lie_batch_in.view(tensor_idx, lie_batch_in.basis());
            op(ctx, tensor, lie, matrix);
        }
    );





}

}// namespace rpp::cpu:::single_thread


#endif //RPP_CPU_OPERATIONS_SINGLE_THREAD_BASIC_LIE_TO_TENSOR_HPP
